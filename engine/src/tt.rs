//! Transposition table: remembers what earlier searches learned about a position.

use shakmaty::Move;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Bound {
    Exact,
    Lower,
    Upper,
}

#[derive(Clone, Copy)]
pub struct Entry {
    pub key: u64,
    pub mv: Option<Move>,
    pub score: i32,
    pub depth: i32,
    pub bound: Bound,
}

pub struct Tt {
    table: Vec<Option<Entry>>,
    mask: usize,
}

impl Tt {
    pub fn new(megabytes: usize) -> Tt {
        let bytes = megabytes.max(1) * 1024 * 1024;
        let slots = (bytes / std::mem::size_of::<Option<Entry>>()).max(1024);
        let size = 1usize << (usize::BITS - 1 - slots.leading_zeros()); // round down to a power of two
        Tt { table: vec![None; size], mask: size - 1 }
    }

    pub fn clear(&mut self) {
        self.table.iter_mut().for_each(|slot| *slot = None);
    }

    pub fn resize(&mut self, megabytes: usize) {
        *self = Tt::new(megabytes);
    }

    pub fn probe(&self, key: u64) -> Option<Entry> {
        self.table[key as usize & self.mask].filter(|entry| entry.key == key)
    }

    pub fn store(&mut self, key: u64, mv: Option<Move>, score: i32, depth: i32, bound: Bound) {
        let slot = &mut self.table[key as usize & self.mask];
        if let Some(old) = slot {
            if old.key == key && old.depth > depth + 2 {
                return; // keep the much deeper result for this same position
            }
            if old.key == key && mv.is_none() {
                *slot = Some(Entry { mv: old.mv, key, score, depth, bound });
                return;
            }
        }
        *slot = Some(Entry { key, mv, score, depth, bound });
    }
}
