//! A small built-in opening book. Lines are written in SAN and stored by position,
//! so the engine finds book moves even when a game reaches a position by a different move order.

use std::collections::HashMap;

use shakmaty::san::San;
use shakmaty::{Chess, Move, Position};

use crate::search::position_key;

const BUILT_IN: &str = include_str!("book.txt");

pub struct Book {
    moves: HashMap<u64, Vec<Move>>,
}

impl Book {
    /// Parse book text: one line of SAN moves per variation, `#` starts a comment line.
    /// Move numbers such as `1.` are ignored. A move written as `~Nf6` is walked through
    /// but never played from the book (an opponent move in a generated repertoire, which
    /// may well be a blunder we only want to punish).
    pub fn parse(text: &str) -> Result<Book, String> {
        let mut moves: HashMap<u64, Vec<Move>> = HashMap::new();
        for (index, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut pos = Chess::default();
            for token in line.split_whitespace() {
                if token.ends_with('.') || token.chars().next().is_some_and(|c| c.is_ascii_digit() && token.contains('.')) {
                    continue;
                }
                let error = |why: String| format!("book line {}: '{token}': {why}", index + 1);
                let (pass_through, san_text) = match token.strip_prefix('~') {
                    Some(rest) => (true, rest),
                    None => (false, token),
                };
                let san = San::from_ascii(san_text.as_bytes()).map_err(|e| error(e.to_string()))?;
                let mv = san.to_move(&pos).map_err(|e| error(e.to_string()))?;
                if !pass_through {
                    moves.entry(position_key(&pos)).or_default().push(mv);
                }
                pos.play_unchecked(mv);
            }
        }
        Ok(Book { moves })
    }

    /// Load a book file, e.g. one written by `openings.py`.
    pub fn load(path: &std::path::Path) -> Result<Book, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        Book::parse(&text)
    }

    pub fn built_in() -> Book {
        Book::parse(BUILT_IN).expect("the built-in book must be valid")
    }

    pub fn positions(&self) -> usize {
        self.moves.len()
    }

    /// A book move for `pos`, chosen with `random` (moves that appear in more lines are likelier).
    pub fn probe(&self, pos: &Chess, random: u64) -> Option<Move> {
        let options = self.moves.get(&position_key(pos))?;
        let choice = options[(random % options.len() as u64) as usize];
        pos.is_legal(choice).then_some(choice)
    }
}
