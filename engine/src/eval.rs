//! Static evaluation: tapered piece-square tables (PeSTO) plus mobility, bishop pair,
//! passed pawns, rook files, king safety and endgame know-how.
//! Scores are centipawns from the side to move's view.

use shakmaty::{attacks, Bitboard, Chess, Color, Piece, Position, Role};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

static KING_SAFETY: AtomicBool = AtomicBool::new(false);
static ENDGAME_RULES: AtomicBool = AtomicBool::new(true);

/// Switch the newer, still-being-tested eval terms on or off (UCI options).
pub fn set_king_safety(on: bool) {
    KING_SAFETY.store(on, Ordering::Relaxed);
}

pub fn set_endgame_rules(on: bool) {
    ENDGAME_RULES.store(on, Ordering::Relaxed);
}

pub fn king_safety_enabled() -> bool {
    KING_SAFETY.load(Ordering::Relaxed)
}

pub fn endgame_rules_enabled() -> bool {
    ENDGAME_RULES.load(Ordering::Relaxed)
}

pub const MATE: i32 = 30_000;
pub const MATE_BOUND: i32 = MATE - 1_000; // scores beyond this mean "forced mate"

const MG_VALUE: [i32; 6] = [82, 337, 365, 477, 1025, 0];
const EG_VALUE: [i32; 6] = [94, 281, 297, 512, 936, 0];
const PHASE_WEIGHT: [i32; 6] = [0, 1, 1, 2, 4, 0];
const TOTAL_PHASE: i32 = 24;

const TEMPO: i32 = 12;
const BISHOP_PAIR: (i32, i32) = (30, 45);
const ROOK_OPEN_FILE: (i32, i32) = (25, 10);
const ROOK_SEMI_OPEN_FILE: (i32, i32) = (12, 8);
const PASSED_MG: [i32; 8] = [0, 0, 5, 10, 20, 35, 60, 0];
const PASSED_EG: [i32; 8] = [0, 5, 10, 25, 45, 80, 130, 0];

const PASSED_FREE_EG: [i32; 8] = [0, 0, 2, 5, 10, 20, 35, 0]; // nothing standing in front
const PASSER_KING_DISTANCE: [i32; 8] = [0, 0, 0, 1, 2, 3, 4, 0]; // kings racing to the pawn
const ENEMY_KING_DISTANCE_WEIGHT: i32 = 5;
const OWN_KING_DISTANCE_WEIGHT: i32 = 2;

// King safety: how dangerous each piece type is when it hits squares around the enemy king.
const KING_ATTACK_WEIGHT: [i32; 4] = [2, 2, 3, 5]; // knight, bishop, rook, queen
const KING_DANGER_DIVISOR: i32 = 3;
const MAX_KING_DANGER: i32 = 400;
const SHIELD_MISSING: i32 = 20;
const SHIELD_ADVANCED: i32 = 10;
const OPEN_FILE_NEAR_KING: i32 = 15;

// Endgame: drive the bare king to the edge, and admit that a lone minor piece can't win.
const MOP_UP_MIN_ADVANTAGE: i32 = 300;
const MOP_UP_EDGE: i32 = 10;
const MOP_UP_CLOSER: i32 = 4;
const NO_WIN_WITHOUT_PAWNS: i32 = 350;
const NO_WIN_DIVISOR: i32 = 8;
const FIFTY_MOVE_SCALE: i32 = 200; // the eval shrinks as the 50-move counter climbs

// Mobility: (baseline squares, middlegame weight, endgame weight) for knight, bishop, rook, queen.
const MOBILITY: [(i32, i32, i32); 4] = [(4, 4, 4), (6, 5, 5), (7, 2, 4), (13, 1, 2)];

const FILE_A: u64 = 0x0101_0101_0101_0101;
const FILE_H: u64 = FILE_A << 7;

// Piece-square tables from white's view, written rank 8 first (index 0 = a8).
#[rustfmt::skip]
const MG_PST: [[i32; 64]; 6] = [
    [ // pawn
          0,   0,   0,   0,   0,   0,  0,   0,
         98, 134,  61,  95,  68, 126, 34, -11,
         -6,   7,  26,  31,  65,  56, 25, -20,
        -14,  13,   6,  21,  23,  12, 17, -23,
        -27,  -2,  -5,  12,  17,   6, 10, -25,
        -26,  -4,  -4, -10,   3,   3, 33, -12,
        -35,  -1, -20, -23, -15,  24, 38, -22,
          0,   0,   0,   0,   0,   0,  0,   0,
    ],
    [ // knight
        -167, -89, -34, -49,  61, -97, -15, -107,
         -73, -41,  72,  36,  23,  62,   7,  -17,
         -47,  60,  37,  65,  84, 129,  73,   44,
          -9,  17,  19,  53,  37,  69,  18,   22,
         -13,   4,  16,  13,  28,  19,  21,   -8,
         -23,  -9,  12,  10,  19,  17,  25,  -16,
         -29, -53, -12,  -3,  -1,  18, -14,  -19,
        -105, -21, -58, -33, -17, -28, -19,  -23,
    ],
    [ // bishop
         -29,   4, -82, -37, -25, -42,   7,  -8,
         -26,  16, -18, -13,  30,  59,  18, -47,
         -16,  37,  43,  40,  35,  50,  37,  -2,
          -4,   5,  19,  50,  37,  37,   7,  -2,
          -6,  13,  13,  26,  34,  12,  10,   4,
           0,  15,  15,  15,  14,  27,  18,  10,
           4,  15,  16,   0,   7,  21,  33,   1,
         -33,  -3, -14, -21, -13, -12, -39, -21,
    ],
    [ // rook
         32,  42,  32,  51, 63,  9,  31,  43,
         27,  32,  58,  62, 80, 67,  26,  44,
         -5,  19,  26,  36, 17, 45,  61,  16,
        -24, -11,   7,  26, 24, 35,  -8, -20,
        -36, -26, -12,  -1,  9, -7,   6, -23,
        -45, -25, -16, -17,  3,  0,  -5, -33,
        -44, -16, -20,  -9, -1, 11,  -6, -71,
        -19, -13,   1,  17, 16,  7, -37, -26,
    ],
    [ // queen
        -28,   0,  29,  12,  59,  44,  43,  45,
        -24, -39,  -5,   1, -16,  57,  28,  54,
        -13, -17,   7,   8,  29,  56,  47,  57,
        -27, -27, -16, -16,  -1,  17,  -2,   1,
         -9, -26,  -9, -10,  -2,  -4,   3,  -3,
        -14,   2, -11,  -2,  -5,   2,  14,   5,
        -35,  -8,  11,   2,   8,  15,  -3,   1,
         -1, -18,  -9,  10, -15, -25, -31, -50,
    ],
    [ // king
        -65,  23,  16, -15, -56, -34,   2,  13,
         29,  -1, -20,  -7,  -8,  -4, -38, -29,
         -9,  24,   2, -16, -20,   6,  22, -22,
        -17, -20, -12, -27, -30, -25, -14, -36,
        -49,  -1, -27, -39, -46, -44, -33, -51,
        -14, -14, -22, -46, -44, -30, -15, -27,
          1,   7,  -8, -64, -43, -16,   9,   8,
        -15,  36,  12, -54,   8, -28,  24,  14,
    ],
];

#[rustfmt::skip]
const EG_PST: [[i32; 64]; 6] = [
    [ // pawn
          0,   0,   0,   0,   0,   0,   0,   0,
        178, 173, 158, 134, 147, 132, 165, 187,
         94, 100,  85,  67,  56,  53,  82,  84,
         32,  24,  13,   5,  -2,   4,  17,  17,
         13,   9,  -3,  -7,  -7,  -8,   3,  -1,
          4,   7,  -6,   1,   0,  -5,  -1,  -8,
         13,   8,   8,  10,  13,   0,   2,  -7,
          0,   0,   0,   0,   0,   0,   0,   0,
    ],
    [ // knight
        -58, -38, -13, -28, -31, -27, -63, -99,
        -25,  -8, -25,  -2,  -9, -25, -24, -52,
        -24, -20,  10,   9,  -1,  -9, -19, -41,
        -17,   3,  22,  22,  22,  11,   8, -18,
        -18,  -6,  16,  25,  16,  17,   4, -18,
        -23,  -3,  -1,  15,  10,  -3, -20, -22,
        -42, -20, -10,  -5,  -2, -20, -23, -44,
        -29, -51, -23, -15, -22, -18, -50, -64,
    ],
    [ // bishop
        -14, -21, -11,  -8, -7,  -9, -17, -24,
         -8,  -4,   7, -12, -3, -13,  -4, -14,
          2,  -8,   0,  -1, -2,   6,   0,   4,
         -3,   9,  12,   9, 14,  10,   3,   2,
         -6,   3,  13,  19,  7,  10,  -3,  -9,
        -12,  -3,   8,  10, 13,   3,  -7, -15,
        -14, -18,  -7,  -1,  4,  -9, -15, -27,
        -23,  -9, -23,  -5, -9, -16,  -5, -17,
    ],
    [ // rook
         13, 10, 18, 15, 12,  12,   8,   5,
         11, 13, 13, 11, -3,   3,   8,   3,
          7,  7,  7,  5,  4,  -3,  -5,  -3,
          4,  3, 13,  1,  2,   1,  -1,   2,
          3,  5,  8,  4, -5,  -6,  -8, -11,
         -4,  0, -5, -1, -7, -12,  -8, -16,
         -6, -6,  0,  2, -9,  -9, -11,  -3,
         -9,  2,  3, -1, -5, -13,   4, -20,
    ],
    [ // queen
         -9,  22,  22,  27,  27,  19,  10,  20,
        -17,  20,  32,  41,  58,  25,  30,   0,
        -20,   6,   9,  49,  47,  35,  19,   9,
          3,  22,  24,  45,  57,  40,  57,  36,
        -18,  28,  19,  47,  31,  34,  39,  23,
        -16, -27,  15,   6,   9,  17,  10,   5,
        -22, -23, -30, -16, -16, -23, -36, -32,
        -33, -28, -22, -43,  -5, -32, -20, -41,
    ],
    [ // king
        -74, -35, -18, -18, -11,  15,   4, -17,
        -12,  17,  14,  17,  17,  38,  23,  11,
         10,  17,  23,  15,  20,  45,  44,  13,
         -8,  22,  24,  27,  26,  33,  26,   3,
        -18,  -4,  21,  24,  27,  23,   9, -11,
        -19,  -3,  11,  21,  23,  16,   7,  -9,
        -27, -11,   4,  13,  14,   4,  -5, -17,
        -53, -34, -21, -11, -28, -14, -24, -43,
    ],
];

struct Tables {
    mg: [[i32; 64]; 6],
    eg: [[i32; 64]; 6],
}

/// Piece-square tables with the piece values already added in.
fn tables() -> &'static Tables {
    static TABLES: OnceLock<Tables> = OnceLock::new();
    TABLES.get_or_init(|| {
        let mut tables = Tables { mg: MG_PST, eg: EG_PST };
        for role in 0..6 {
            for square in 0..64 {
                tables.mg[role][square] += MG_VALUE[role];
                tables.eg[role][square] += EG_VALUE[role];
            }
        }
        tables
    })
}

fn adjacent_files(file: usize) -> u64 {
    let mut mask = FILE_A << file;
    if file > 0 {
        mask |= FILE_A << (file - 1);
    }
    if file < 7 {
        mask |= FILE_A << (file + 1);
    }
    mask
}

fn is_passed_pawn(square: usize, is_white: bool, enemy_pawns: u64) -> bool {
    let (file, rank) = (square & 7, square >> 3);
    let ahead = if is_white {
        if rank == 7 { 0 } else { !0u64 << (8 * (rank + 1)) }
    } else if rank == 0 {
        0
    } else {
        (1u64 << (8 * rank)) - 1
    };
    enemy_pawns & adjacent_files(file) & ahead == 0
}

fn pawn_attacks(pawns: u64, is_white: bool) -> u64 {
    if is_white {
        ((pawns << 7) & !FILE_H) | ((pawns << 9) & !FILE_A)
    } else {
        ((pawns >> 7) & !FILE_A) | ((pawns >> 9) & !FILE_H)
    }
}

/// True when neither side has pieces beyond kings and pawns (null-move search is unsafe).
pub fn is_pawn_ending(pos: &Chess, side: Color) -> bool {
    let board = pos.board();
    (board.by_color(side) & !(board.pawns() | board.kings())).is_empty()
}

fn distance(a: usize, b: usize) -> i32 {
    let files = (a & 7) as i32 - (b & 7) as i32;
    let ranks = (a >> 3) as i32 - (b >> 3) as i32;
    files.abs().max(ranks.abs())
}

fn distance_from_center(square: usize) -> i32 {
    let (file, rank) = ((square & 7) as i32, (square >> 3) as i32);
    (2 * file - 7).abs() / 2 + (2 * rank - 7).abs() / 2
}

/// The king, the squares next to it and one more rank in front.
fn king_zone(king: usize, is_white: bool) -> u64 {
    let near = u64::from(attacks::king_attacks(shakmaty::Square::new(king as u32))) | (1u64 << king);
    let forward = if is_white { near << 8 } else { near >> 8 };
    near | forward
}

/// Middlegame penalty for missing or pushed pawns in front of a castled-looking king.
fn pawn_shield_penalty(king: usize, is_white: bool, own_pawns: u64, all_pawns: u64) -> i32 {
    let relative_rank = if is_white { king >> 3 } else { 7 - (king >> 3) };
    if relative_rank > 1 {
        return 0;
    }
    let king_file = king & 7;
    let mut penalty = 0;
    for file in king_file.saturating_sub(1)..=(king_file + 1).min(7) {
        let on_file = FILE_A << file;
        let shield = own_pawns & on_file;
        if shield == 0 {
            penalty += SHIELD_MISSING;
            if all_pawns & on_file == 0 {
                penalty += OPEN_FILE_NEAR_KING;
            }
            continue;
        }
        let nearest = if is_white { shield.trailing_zeros() as usize } else { 63 - shield.leading_zeros() as usize };
        let rank_gap = ((nearest >> 3) as i32 - (king >> 3) as i32).abs();
        if rank_gap > 2 {
            penalty += SHIELD_ADVANCED;
        }
    }
    penalty
}

pub fn evaluate(pos: &Chess) -> i32 {
    let king_safety = king_safety_enabled();
    let endgame_rules = endgame_rules_enabled();
    let board = pos.board();
    let tables = tables();
    let occupied = board.occupied();

    let pawns = [
        u64::from(board.by_piece(Piece { color: Color::White, role: Role::Pawn })),
        u64::from(board.by_piece(Piece { color: Color::Black, role: Role::Pawn })),
    ];
    let own = [board.white(), board.black()];
    let enemy_pawn_attacks = [pawn_attacks(pawns[1], false), pawn_attacks(pawns[0], true)];

    let mut mg = [0i32; 2];
    let mut eg = [0i32; 2];
    let mut bishops = [0i32; 2];
    let mut phase = 0i32;
    let kings = [
        board.king_of(Color::White).map_or(0, |sq| sq as usize),
        board.king_of(Color::Black).map_or(0, |sq| sq as usize),
    ];
    let zones = [king_zone(kings[0], true), king_zone(kings[1], false)];
    let mut king_attackers = [0i32; 2]; // pieces of `side` hitting the enemy king zone
    let mut king_attack_units = [0i32; 2];
    let mut piece_material = [0i32; 2];

    for square in occupied {
        let Some(piece) = board.piece_at(square) else { continue };
        let side = if piece.color.is_white() { 0 } else { 1 };
        let role = piece.role as usize - 1;
        let sq = square as usize;
        let table_index = if side == 0 { sq ^ 56 } else { sq };
        mg[side] += tables.mg[role][table_index];
        eg[side] += tables.eg[role][table_index];
        phase += PHASE_WEIGHT[role];

        match piece.role {
            Role::Pawn => {
                if is_passed_pawn(sq, side == 0, pawns[1 - side]) {
                    let relative_rank = if side == 0 { sq >> 3 } else { 7 - (sq >> 3) };
                    mg[side] += PASSED_MG[relative_rank];
                    eg[side] += PASSED_EG[relative_rank];
                    let stop = if side == 0 { sq + 8 } else { sq.wrapping_sub(8) };
                    if endgame_rules && stop < 64 {
                        if u64::from(occupied) & (1u64 << stop) == 0 {
                            eg[side] += PASSED_FREE_EG[relative_rank];
                        }
                        let race = ENEMY_KING_DISTANCE_WEIGHT * distance(kings[1 - side], stop)
                            - OWN_KING_DISTANCE_WEIGHT * distance(kings[side], stop);
                        eg[side] += race * PASSER_KING_DISTANCE[relative_rank];
                    }
                }
            }
            Role::Knight | Role::Bishop | Role::Rook | Role::Queen => {
                if piece.role == Role::Bishop {
                    bishops[side] += 1;
                }
                piece_material[side] += EG_VALUE[role];
                let hits = attacks::attacks(square, piece, occupied);
                let zone_hits = (u64::from(hits) & zones[1 - side]).count_ones() as i32;
                if king_safety && zone_hits > 0 {
                    king_attackers[side] += 1;
                    king_attack_units[side] += zone_hits * KING_ATTACK_WEIGHT[role - 1];
                }
                let reachable = hits
                    & !own[side]
                    & !Bitboard::from(enemy_pawn_attacks[side]);
                let (baseline, mg_weight, eg_weight) = MOBILITY[role - 1];
                let moves = reachable.count() as i32 - baseline;
                mg[side] += moves * mg_weight;
                eg[side] += moves * eg_weight;
                if piece.role == Role::Rook {
                    let file_mask = FILE_A << (sq & 7);
                    let (open, semi) = (ROOK_OPEN_FILE, ROOK_SEMI_OPEN_FILE);
                    if (pawns[0] | pawns[1]) & file_mask == 0 {
                        mg[side] += open.0;
                        eg[side] += open.1;
                    } else if pawns[side] & file_mask == 0 {
                        mg[side] += semi.0;
                        eg[side] += semi.1;
                    }
                }
            }
            Role::King => {}
        }
    }

    for side in 0..2 {
        if bishops[side] >= 2 {
            mg[side] += BISHOP_PAIR.0;
            eg[side] += BISHOP_PAIR.1;
        }
        if king_attackers[side] >= 2 {
            let units = king_attack_units[side];
            mg[side] += (units * units / KING_DANGER_DIVISOR).min(MAX_KING_DANGER);
        }
        if king_safety {
            mg[side] -= pawn_shield_penalty(kings[side], side == 0, pawns[side], pawns[0] | pawns[1]);
        }
    }

    let phase = phase.min(TOTAL_PHASE);
    let mg_score = mg[0] - mg[1];
    let mut eg_score = eg[0] - eg[1];
    if endgame_rules {
        eg_score += endgame_adjustment(eg_score, kings, pawns, piece_material);
    }
    let mut white_view = (mg_score * phase + eg_score * (TOTAL_PHASE - phase)) / TOTAL_PHASE;
    if endgame_rules {
        white_view = white_view * (FIFTY_MOVE_SCALE - pos.halfmoves().min(100) as i32) / FIFTY_MOVE_SCALE;
    }
    let from_side_to_move = if pos.turn().is_white() { white_view } else { -white_view };
    from_side_to_move + TEMPO
}

/// Endgame corrections to the white-view endgame score.
fn endgame_adjustment(eg_score: i32, kings: [usize; 2], pawns: [u64; 2], piece_material: [i32; 2]) -> i32 {
    let (strong, weak) = if eg_score >= 0 { (0, 1) } else { (1, 0) };
    let sign = if strong == 0 { 1 } else { -1 };
    // a lone minor piece (or less) without pawns can't force mate
    if pawns[strong] == 0 && piece_material[strong] - piece_material[weak] <= NO_WIN_WITHOUT_PAWNS {
        return -eg_score + eg_score / NO_WIN_DIVISOR;
    }
    // winning against a bare king: push it to the edge and bring ours closer
    if pawns[weak] == 0 && eg_score.abs() >= MOP_UP_MIN_ADVANTAGE {
        let edge = distance_from_center(kings[weak]) * MOP_UP_EDGE;
        let closer = (14 - 2 * distance(kings[0], kings[1])) * MOP_UP_CLOSER;
        return sign * (edge + closer);
    }
    0
}
