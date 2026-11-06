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
    /// Move numbers such as `1.` are ignored.
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
                let san = San::from_ascii(token.as_bytes()).map_err(|e| error(e.to_string()))?;
                let mv = san.to_move(&pos).map_err(|e| error(e.to_string()))?;
                moves.entry(position_key(&pos)).or_default().push(mv);
                pos.play_unchecked(mv);
            }
        }
        Ok(Book { moves })
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
