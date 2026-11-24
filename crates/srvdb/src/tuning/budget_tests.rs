use super::*;

const MIB: u64 = 1 << 20;

#[test]
fn layouts_match_adr_0018() {
    // 384 dims: 1536 B flat, 192 + 8 B at 4 bits, 96 + 8 at 2, 48 + 8 at 1
    assert_eq!(tier_vector_bytes(Tier::Flat, 384, 32), 32 * 1536);
    assert_eq!(tier_vector_bytes(Tier::Rabitq(4), 384, 32), 32 * 200);
    assert_eq!(tier_vector_bytes(Tier::Rabitq(2), 384, 32), 32 * 104);
    assert_eq!(tier_vector_bytes(Tier::Rabitq(1), 384, 32), 32 * 56);
    // Codes are padded to whole 32-vector blocks, factors and widths not
    assert_eq!(
        tier_vector_bytes(Tier::Rabitq(1), 384, 33),
        64 * 48 + 33 * 8
    );
    // Odd dimensions round the code up to a whole byte
    assert_eq!(rabitq_code_bytes(5, 1), 1);
    assert_eq!(rabitq_code_bytes(5, 4), 3);
    assert_eq!(tier_fixed_bytes(Tier::Rabitq(2), 384), 4 * 384 * 384);
    assert_eq!(tier_fixed_bytes(Tier::Flat, 384), 0);
    assert_eq!(sq8_bytes(384, 10), (3840, 8 * 384));
    assert_eq!(pq_bytes(384, 48, 10), (480, 1024 * 384));
    assert_eq!(hnsw_bytes(16, 10), 10 * (256 + 96));
    assert_eq!(limit_bytes(10), 10 * MIB / 100 * 80);
}

#[test]
fn selection_is_exact_at_every_boundary() {
    let dim = 128;
    let limit = limit_bytes(4);
    for (i, tier) in LADDER.into_iter().enumerate() {
        let cap = capacity(tier, dim, 0, limit);
        // At capacity the tier still fits; one more record does not
        assert!(tier_bytes(tier, dim, cap) <= limit);
        assert!(tier_bytes(tier, dim, cap + 1) > limit);
        assert_eq!(select(Tier::Flat, dim, cap, 0, limit), Some(tier));
        assert_eq!(
            select(Tier::Flat, dim, cap + 1, 0, limit),
            LADDER.get(i + 1).copied(),
            "past {tier:?}"
        );
    }
}

#[test]
fn other_resident_bytes_shrink_what_fits() {
    let limit = limit_bytes(1);
    let alone = capacity(Tier::Flat, 64, 0, limit);
    let shared = capacity(Tier::Flat, 64, limit / 2, limit);
    assert!(shared < alone);
    assert_eq!(capacity(Tier::Flat, 64, limit + 1, limit), 0);
    assert_eq!(select(Tier::Flat, 64, 0, limit + 1, limit), None);
}

#[test]
fn a_rotation_larger_than_the_budget_keeps_flat_only() {
    // 4096 dims: the rotation alone is 64 MiB
    let limit = limit_bytes(16);
    assert_eq!(select(Tier::Flat, 4096, 0, 0, limit), Some(Tier::Flat));
    let cap = capacity(Tier::Flat, 4096, 0, limit);
    assert_eq!(select(Tier::Flat, 4096, cap + 1, 0, limit), None);
}

#[test]
fn huge_budgets_do_not_overflow() {
    let limit = limit_bytes(MAX_MEMORY_MB);
    assert!(capacity(Tier::Rabitq(1), 1, 0, limit) > 0);
    assert_eq!(limit_bytes(u64::MAX), u64::MAX / 100 * HEADROOM_PERCENT);
}
