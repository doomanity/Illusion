//! Static exchange evaluation: does a capture win or lose material if both sides keep trading?

use shakmaty::{Bitboard, Board, Move, Role, Square};

/// Piece values in centipawns, indexed by `Role as usize` (Pawn = 1 .. King = 6).
pub const VALUES: [i32; 7] = [0, 100, 320, 330, 500, 900, 20_000];

pub fn role_value(role: Role) -> i32 {
    VALUES[role as usize]
}

fn least_valuable_attacker(board: &Board, attackers: Bitboard) -> Option<(Square, Role)> {
    Role::ALL.into_iter().find_map(|role| {
        (attackers & board.by_role(role)).first().map(|sq| (sq, role))
    })
}

/// Net material the mover wins (negative = loses) by playing `m` and trading on its target square.
pub fn see(board: &Board, m: Move) -> i32 {
    if m.is_castle() {
        return 0;
    }
    let Some(from) = m.from() else { return 0 };
    let Some(mover) = board.color_at(from) else { return 0 };
    let to = m.to();

    let mut occupied = board.occupied() ^ Bitboard::from_square(from);
    let mut gain = [0i32; 32];
    gain[0] = if m.is_en_passant() {
        occupied ^= Bitboard::from_square(Square::from_coords(to.file(), from.rank()));
        role_value(Role::Pawn)
    } else {
        m.capture().map_or(0, role_value)
    };
    let mut piece_on_target = match m.promotion() {
        Some(promoted) => {
            gain[0] += role_value(promoted) - role_value(Role::Pawn);
            role_value(promoted)
        }
        None => role_value(m.role()),
    };

    let mut side = mover.other();
    let mut depth = 0usize;
    loop {
        depth += 1;
        if depth >= gain.len() {
            break;
        }
        gain[depth] = piece_on_target - gain[depth - 1];
        let attackers = board.attacks_to(to, side, occupied) & occupied;
        let Some((square, role)) = least_valuable_attacker(board, attackers) else { break };
        occupied ^= Bitboard::from_square(square);
        piece_on_target = role_value(role);
        side = side.other();
    }
    // the last entry was speculative (no capture happened), so unwind from the one before it
    while depth > 1 {
        depth -= 1;
        gain[depth - 1] = -(-gain[depth - 1]).max(gain[depth]);
    }
    gain[0]
}
