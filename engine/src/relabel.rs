//! Teacher labels: rescore recorded positions with a stronger UCI engine (e.g. Stockfish)
//! while keeping each position's game result.
//!
//! Output records match input records one to one and in order, so a stopped run resumes
//! by skipping as many input records as the output already holds.

use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::time::Instant;

use shakmaty::Color;

use crate::datagen::{Sample, SAMPLE_BYTES};

const CHUNK_POSITIONS: usize = 2048;
const TEACHER_HASH_MB: u32 = 16;
/// Mate scores and anything beyond this are stored as this many centipawns.
pub const MAX_TEACHER_SCORE: i32 = 3000;
const PROGRESS_EVERY_POSITIONS: usize = 100_000;

/// A UCI engine running as a child process.
pub struct UciEngine {
    child: Child,
    input: ChildStdin,
    output: BufReader<ChildStdout>,
}

impl UciEngine {
    pub fn start(path: &Path) -> io::Result<UciEngine> {
        let mut child = Command::new(path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| io::Error::new(e.kind(), format!("cannot start {}: {e}", path.display())))?;
        let input = child.stdin.take().expect("piped stdin");
        let output = BufReader::new(child.stdout.take().expect("piped stdout"));
        let mut engine = UciEngine { child, input, output };
        engine.send("uci")?;
        engine.wait_for("uciok")?;
        engine.send("setoption name Threads value 1")?;
        engine.send(&format!("setoption name Hash value {TEACHER_HASH_MB}"))?;
        engine.send("isready")?;
        engine.wait_for("readyok")?;
        Ok(engine)
    }

    fn send(&mut self, command: &str) -> io::Result<()> {
        writeln!(self.input, "{command}")?;
        self.input.flush()
    }

    fn read_line(&mut self) -> io::Result<String> {
        let mut line = String::new();
        if self.output.read_line(&mut line)? == 0 {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "teacher engine quit"));
        }
        Ok(line)
    }

    fn wait_for(&mut self, token: &str) -> io::Result<()> {
        while !self.read_line()?.trim_start().starts_with(token) {}
        Ok(())
    }

    /// Score of `fen` in centipawns for the side to move, searched to `nodes`.
    pub fn score(&mut self, fen: &str, nodes: u64) -> io::Result<i32> {
        self.send(&format!("position fen {fen}"))?;
        self.send(&format!("go nodes {nodes}"))?;
        let mut score = None;
        loop {
            let line = self.read_line()?;
            if line.starts_with("bestmove") {
                break;
            }
            if let Some(found) = parse_score(&line) {
                score = Some(found);
            }
        }
        score.ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "teacher gave no score"))
    }
}

impl Drop for UciEngine {
    fn drop(&mut self) {
        let _ = self.send("quit");
        let _ = self.child.wait();
    }
}

/// The score in an `info` line, clipped to `MAX_TEACHER_SCORE` (mates become the clip value).
pub fn parse_score(line: &str) -> Option<i32> {
    let mut words = line.split_whitespace();
    if words.next() != Some("info") {
        return None;
    }
    while let Some(word) = words.next() {
        if word != "score" {
            continue;
        }
        let kind = words.next()?;
        let value: i32 = words.next()?.parse().ok()?;
        return match kind {
            "cp" => Some(value.clamp(-MAX_TEACHER_SCORE, MAX_TEACHER_SCORE)),
            "mate" if value > 0 => Some(MAX_TEACHER_SCORE),
            "mate" => Some(-MAX_TEACHER_SCORE),
            _ => None,
        };
    }
    None
}

/// FEN for a sample. Castling rights and en passant are not recorded, so they are left out.
pub fn sample_fen(sample: &Sample) -> String {
    let mut board = shakmaty::Board::empty();
    for (square, piece) in sample.pieces() {
        board.set_piece_at(square, piece);
    }
    let side = if sample.side_to_move == Color::White { 'w' } else { 'b' };
    format!("{board} {side} - - 0 1")
}

fn relabel_chunk(engine: &mut UciEngine, records: &[u8], nodes: u64) -> io::Result<Vec<u8>> {
    let mut out = Vec::with_capacity(records.len());
    for record in records.chunks_exact(SAMPLE_BYTES) {
        let mut sample = Sample::from_bytes(record).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let stm_score = engine.score(&sample_fen(&sample), nodes)?;
        let white_score = if sample.side_to_move == Color::White { stm_score } else { -stm_score };
        sample.score = white_score as i16;
        out.extend_from_slice(&sample.to_bytes());
    }
    Ok(out)
}

#[derive(Clone, Debug)]
pub struct RelabelConfig {
    pub teacher: PathBuf,
    pub nodes: u64,
    pub threads: usize,
}

/// Rescore every position of `input` into `output`, resuming if `output` already has some.
pub fn relabel(input: &Path, output: &Path, config: &RelabelConfig) -> io::Result<()> {
    let data = fs::read(input)?;
    let total = data.len() / SAMPLE_BYTES;
    let done_bytes = fs::metadata(output).map_or(0, |m| m.len() as usize);
    let already = (done_bytes / SAMPLE_BYTES).min(total);
    let file = OpenOptions::new().create(true).write(true).truncate(false).open(output)?;
    file.set_len((already * SAMPLE_BYTES) as u64)?; // drop a half-written record, if any
    let mut writer = BufWriter::new(OpenOptions::new().append(true).open(output)?);
    drop(file);
    println!(
        "rescoring {} positions with {} at {} nodes on {} threads ({} already done)",
        total - already,
        config.teacher.display(),
        config.nodes,
        config.threads,
        already
    );

    let todo = &data[already * SAMPLE_BYTES..total * SAMPLE_BYTES];
    let chunks: Vec<&[u8]> = todo.chunks(CHUNK_POSITIONS * SAMPLE_BYTES).collect();
    let next_chunk = AtomicUsize::new(0);
    let start = Instant::now();
    let (sender, receiver) = mpsc::channel::<io::Result<(usize, Vec<u8>)>>();

    std::thread::scope(|scope| -> io::Result<()> {
        for _ in 0..config.threads.max(1) {
            let (sender, next_chunk, chunks) = (sender.clone(), &next_chunk, &chunks);
            scope.spawn(move || {
                let mut engine = match UciEngine::start(&config.teacher) {
                    Ok(engine) => engine,
                    Err(error) => return drop(sender.send(Err(error))),
                };
                loop {
                    let index = next_chunk.fetch_add(1, Ordering::Relaxed);
                    let Some(chunk) = chunks.get(index) else { return };
                    let result = relabel_chunk(&mut engine, chunk, config.nodes).map(|bytes| (index, bytes));
                    let failed = result.is_err();
                    if sender.send(result).is_err() || failed {
                        return;
                    }
                }
            });
        }
        drop(sender);

        // write chunks in input order so the output always lines up with the input
        let mut pending = BTreeMap::new();
        let (mut next_to_write, mut written) = (0usize, 0usize);
        for message in receiver {
            let (index, bytes) = match message {
                Ok(done) => done,
                Err(error) => {
                    next_chunk.store(usize::MAX / 2, Ordering::Relaxed); // stop the other threads
                    return Err(error);
                }
            };
            pending.insert(index, bytes);
            while let Some(bytes) = pending.remove(&next_to_write) {
                writer.write_all(&bytes)?;
                writer.flush()?;
                let before = written;
                written += bytes.len() / SAMPLE_BYTES;
                next_to_write += 1;
                if written / PROGRESS_EVERY_POSITIONS > before / PROGRESS_EVERY_POSITIONS {
                    let rate = written as f64 / start.elapsed().as_secs_f64().max(0.001);
                    let left_hours = (total - already - written) as f64 / rate / 3600.0;
                    println!("{} / {} positions, {rate:.0} per second, about {left_hours:.1} h left", already + written, total);
                }
            }
        }
        Ok(())
    })?;
    println!("done in {:.0}s, output {}", start.elapsed().as_secs_f64(), output.display());
    Ok(())
}
