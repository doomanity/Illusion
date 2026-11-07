use std::path::{Path, PathBuf};
use std::process::ExitCode;

use illusion_engine::datagen::{self, DatagenConfig};
use illusion_engine::nnue::{self, Network};
use illusion_engine::train::{self, TrainConfig};
use illusion_engine::uci;

const USAGE: &str = "usage:
  illusion                                   run as a UCI engine
  illusion bench [depth] [--net file.nnue]   speed test
  illusion datagen <out.bin> [--games N] [--threads N] [--nodes N] [--net file.nnue]
  illusion train <out.nnue> <data.bin>... [--epochs N] [--threads N]";

/// Positional arguments and `--name value` flags.
struct Args {
    positional: Vec<String>,
    flags: Vec<(String, String)>,
}

impl Args {
    fn parse(raw: &[String]) -> Result<Args, String> {
        let (mut positional, mut flags) = (Vec::new(), Vec::new());
        let mut iter = raw.iter();
        while let Some(arg) = iter.next() {
            match arg.strip_prefix("--") {
                Some(name) => {
                    let value = iter.next().ok_or(format!("--{name} needs a value"))?;
                    flags.push((name.to_string(), value.clone()));
                }
                None => positional.push(arg.clone()),
            }
        }
        Ok(Args { positional, flags })
    }

    fn flag<T: std::str::FromStr>(&self, name: &str, default: T) -> Result<T, String> {
        match self.flags.iter().rev().find(|(n, _)| n == name) {
            Some((_, value)) => value.parse().map_err(|_| format!("bad value for --{name}: {value}")),
            None => Ok(default),
        }
    }

    fn text_flag(&self, name: &str) -> Option<&str> {
        self.flags.iter().rev().find(|(n, _)| n == name).map(|(_, v)| v.as_str())
    }
}

fn default_threads() -> usize {
    std::thread::available_parallelism().map_or(4, |n| n.get())
}

fn datagen_command(args: &Args) -> Result<(), String> {
    let out = args.positional.first().ok_or("datagen needs an output file")?;
    if let Some(net) = args.text_flag("net") {
        nnue::set_active(Some(Network::load(Path::new(net))?));
    }
    let config = DatagenConfig {
        games: args.flag("games", 100_000)?,
        threads: args.flag("threads", default_threads().saturating_sub(1).max(1))?,
        nodes: args.flag("nodes", 5_000)?,
    };
    datagen::generate(Path::new(out), config).map_err(|e| e.to_string())
}

fn train_command(args: &Args) -> Result<(), String> {
    let (out, data) = args.positional.split_first().ok_or("train needs an output file and data files")?;
    if data.is_empty() {
        return Err("train needs at least one data file".into());
    }
    let defaults = TrainConfig::default();
    let config = TrainConfig {
        epochs: args.flag("epochs", defaults.epochs)?,
        threads: args.flag("threads", defaults.threads)?,
        ..defaults
    };
    let files: Vec<PathBuf> = data.iter().map(PathBuf::from).collect();
    let samples = train::load_samples(&files)?;
    if samples.len() < 1000 {
        return Err(format!("only {} positions, make more data first", samples.len()));
    }
    train::train(samples, Path::new(out), &config).map(|_| ()).map_err(|e| e.to_string())
}

fn run(raw: &[String]) -> Result<(), String> {
    let Some(command) = raw.first() else {
        uci::run();
        return Ok(());
    };
    let args = Args::parse(&raw[1..])?;
    match command.as_str() {
        "bench" => {
            if let Some(net) = args.text_flag("net") {
                nnue::set_active(Some(Network::load(Path::new(net))?));
            }
            let depth = args.positional.first().and_then(|d| d.parse().ok()).unwrap_or(9);
            let (nodes, millis) = uci::bench(depth);
            println!("{nodes} nodes {millis} ms {} nps", nodes as u128 * 1000 / millis.max(1));
            Ok(())
        }
        "datagen" => datagen_command(&args),
        "train" => train_command(&args),
        "help" | "--help" | "-h" => {
            println!("{USAGE}");
            Ok(())
        }
        other => Err(format!("unknown command '{other}'\n{USAGE}")),
    }
}

fn main() -> ExitCode {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    match run(&raw) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("{message}");
            ExitCode::FAILURE
        }
    }
}
