//! Human mode playing style: love of attacking the enemy king, trading down when ahead,
//! keeping pieces on when behind, and refusing cheap draws.
//!
//! All of it is one-sided: it only rewards the engine's own side (`me`), so the engine
//! plays this way itself instead of expecting the same of its opponent.

use shakmaty::{attacks, Board, Chess, Color, Position, Role};

use crate::eval::{king_zone, KING_ATTACK_WEIGHT};

pub const DEFAULT_AGGRESSION: i32 = 50;
pub const DEFAULT_CONTEMPT: i32 = 25;
pub const MAX_AGGRESSION: i32 = 100;
pub const MAX_CONTEMPT: i32 = 200;

/// Centipawns per unit of pressure on the enemy king at full aggression.
const ATTACK_CP_PER_UNIT: i32 = 4;
const MAX_ATTACK_BONUS: i32 = 120;
/// Centipawns per traded piece when clearly winning (and the reverse when losing).
const TRADE_CP_PER_PIECE: i32 = 6;
/// Score at which the trading urge is at full strength.
const TRADE_FULL_AT: i32 = 300;
const STARTING_PIECES: i32 = 14; // knights, bishops, rooks and queens of both sides

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Style {
    /// 0 to 100: how much attacking the enemy king is worth.
    pub aggression: i32,
    /// Centipawns a draw is worth less than zero to us.
    pub contempt: i32,
}

impl Default for Style {
    fn default() -> Self {
        Style { aggression: DEFAULT_AGGRESSION, contempt: DEFAULT_CONTEMPT }
    }
}

/// Weighted count of `attacker`'s knight, bishop, rook and queen hits on squares around
/// the other king.
pub fn king_pressure(board: &Board, attacker: Color) -> i32 {
    let Some(king) = board.king_of(!attacker) else { return 0 };
    let zone = king_zone(king as usize, (!attacker).is_white());
    let occupied = board.occupied();
    let mut units = 0;
    for (i, role) in [Role::Knight, Role::Bishop, Role::Rook, Role::Queen].into_iter().enumerate() {
        for square in board.by_piece(role.of(attacker)) {
            let hits = attacks::attacks(square, role.of(attacker), occupied);
            units += (u64::from(hits) & zone).count_ones() as i32 * KING_ATTACK_WEIGHT[i];
        }
    }
    units
}

fn pieces_left(board: &Board) -> i32 {
    (board.knights() | board.bishops() | board.rooks() | board.queens()).count() as i32
}

/// Style bonus for `me`, in centipawns from `me`'s point of view. `my_score` is the plain
/// evaluation from `me`'s point of view (it decides whether trades are welcome).
pub fn bonus_for(board: &Board, me: Color, my_score: i32, style: Style) -> i32 {
    let attack = (king_pressure(board, me) * ATTACK_CP_PER_UNIT * style.aggression / MAX_AGGRESSION).min(MAX_ATTACK_BONUS);
    let traded = STARTING_PIECES - pieces_left(board);
    let urge = my_score.clamp(-TRADE_FULL_AT, TRADE_FULL_AT);
    let trades = traded * TRADE_CP_PER_PIECE * urge / TRADE_FULL_AT;
    attack + trades
}

/// `score` (side to move's view) with the style bonus for `me` added.
pub fn adjust(pos: &Chess, score: i32, me: Color, style: Style) -> i32 {
    let mine = if pos.turn() == me { score } else { -score };
    let bonus = bonus_for(pos.board(), me, mine, style);
    if pos.turn() == me { score + bonus } else { score - bonus }
}

/// What a draw is worth to the side to move, given that the engine plays `me`.
pub fn draw_score(pos: &Chess, me: Color, style: Style) -> i32 {
    if pos.turn() == me { -style.contempt } else { style.contempt }
}
