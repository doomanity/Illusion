mod common;

use common::{pos, START};
use illusion_engine::style::{adjust, bonus_for, draw_score, king_pressure, Style};
use illusion_engine::swindle::{effective_tolerance, Settings};
use shakmaty::{Color, Position};

#[test]
fn nobody_is_attacking_in_the_start_position() {
    let start = pos(START);
    assert_eq!(king_pressure(start.board(), Color::White), 0);
    assert_eq!(king_pressure(start.board(), Color::Black), 0);
}

#[test]
fn pieces_aimed_at_the_king_create_pressure() {
    let attack = pos("6k1/5ppp/8/5N2/6Q1/8/5PPP/6K1 w - - 0 1");
    assert!(king_pressure(attack.board(), Color::White) > 5);
    assert_eq!(king_pressure(attack.board(), Color::Black), 0);
}

#[test]
fn more_aggression_means_a_bigger_attack_bonus() {
    let attack = pos("6k1/5ppp/8/5N2/6Q1/8/5PPP/6K1 w - - 0 1");
    let calm = bonus_for(attack.board(), Color::White, 0, Style { aggression: 0, contempt: 0 });
    let wild = bonus_for(attack.board(), Color::White, 0, Style { aggression: 100, contempt: 0 });
    assert_eq!(calm, 0);
    assert!(wild > 20, "{wild}");
}

#[test]
fn trades_are_welcome_when_winning_and_avoided_when_losing() {
    let few_pieces = pos("4k3/pppp4/8/8/8/8/PPPP4/3RK3 w - - 0 1");
    let style = Style { aggression: 0, contempt: 0 };
    assert!(bonus_for(few_pieces.board(), Color::White, 400, style) > 0);
    assert!(bonus_for(few_pieces.board(), Color::White, -400, style) < 0);
    assert_eq!(bonus_for(few_pieces.board(), Color::White, 0, style), 0);
}

#[test]
fn style_only_helps_our_own_side() {
    // same board, either side to move: the bonus must count for us both times
    let style = Style { aggression: 100, contempt: 0 };
    let ours = pos("6k1/5ppp/8/5N2/6Q1/8/5PPP/6K1 w - - 0 1");
    let theirs = pos("6k1/5ppp/8/5N2/6Q1/8/5PPP/6K1 b - - 0 1");
    let gain_when_we_move = adjust(&ours, 0, Color::White, style);
    let gain_when_they_move = -adjust(&theirs, 0, Color::White, style);
    assert!(gain_when_we_move > 0);
    assert_eq!(gain_when_we_move, gain_when_they_move);
}

#[test]
fn draws_are_bad_for_us_and_good_for_the_opponent() {
    let style = Style { aggression: 0, contempt: 30 };
    let start = pos(START);
    assert_eq!(draw_score(&start, Color::White, style), -30);
    assert_eq!(draw_score(&start, Color::Black, style), 30);
}

#[test]
fn traps_get_less_room_against_stronger_opponents() {
    let at = |elo| effective_tolerance(Settings { opponent_elo: elo, tolerance: 60, ..Settings::default() });
    assert_eq!(at(1200), 60);
    assert!(at(2200) < at(1900));
    assert_eq!(at(2600), 0);
    assert_eq!(at(3000), 0);
}
