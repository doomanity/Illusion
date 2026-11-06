#![allow(dead_code)]
use shakmaty::fen::Fen;
use shakmaty::{CastlingMode, Chess, Move, Position};

pub fn pos(fen: &str) -> Chess {
    Fen::from_ascii(fen.as_bytes()).unwrap().into_position(CastlingMode::Standard).unwrap()
}

pub fn uci(m: Move) -> String {
    m.to_uci(CastlingMode::Standard).to_string()
}

pub fn is_legal_uci(position: &Chess, text: &str) -> bool {
    position.legal_moves().iter().any(|&m| uci(m) == text)
}

pub const START: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
