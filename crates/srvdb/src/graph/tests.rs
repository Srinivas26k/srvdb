use super::expand::expand;
use super::*;
use crate::SearchResult;

#[test]
fn type_names_follow_the_allow_pattern() {
    for ok in ["cites", "a", "part_of", "v2", &"x".repeat(64)] {
        assert!(validate_type(ok).is_ok(), "{ok}");
    }
    for bad in [
        "",
        "Cites",
        "1st",
        "_x",
        "has space",
        "a-b",
        "é",
        &"x".repeat(65),
    ] {
        assert!(validate_type(bad).is_err(), "{bad}");
    }
}

#[test]
fn add_remove_and_duplicates() {
    let mut g = LinkGraph::default();
    assert!(g.add(1, "cites", 2).unwrap());
    assert!(!g.add(1, "cites", 2).unwrap());
    assert!(g.add(1, "about", 2).unwrap());
    assert_eq!(g.len(), 2);
    assert_eq!(g.outgoing(1, Some("cites")), vec![("cites".into(), 2)]);
    assert!(g.remove(1, "cites", 2));
    assert!(!g.remove(1, "cites", 2));
    assert_eq!(g.len(), 1);
    assert_eq!(g.pending().len(), 3);
}

#[test]
fn fan_out_is_bounded() {
    let mut g = LinkGraph::default();
    for t in 0..MAX_LINKS_PER_RECORD as u64 {
        g.add(0, "x", t + 1).unwrap();
    }
    assert!(g.add(0, "x", 999_999).is_err());
    // Incoming links are not limited
    assert!(g.add(5000, "x", 1).is_ok());
}

#[test]
fn removing_a_record_drops_both_directions() {
    let mut g = LinkGraph::default();
    g.add(1, "a", 2).unwrap();
    g.add(3, "a", 1).unwrap();
    g.add(3, "a", 2).unwrap();
    g.remove_record(1);
    assert_eq!(g.all(), vec![(3, "a".into(), 2)]);
    assert!(!g.inc.contains_key(&1) && !g.out.contains_key(&1));
}

#[test]
fn moving_a_record_keeps_its_links() {
    let mut g = LinkGraph::default();
    g.add(1, "a", 2).unwrap();
    g.add(3, "a", 1).unwrap();
    g.add(1, "a", 9).unwrap();
    g.move_record(1, 9).unwrap();
    // 1 -> 9 would become a self-link and is dropped
    assert_eq!(g.all(), vec![(3, "a".into(), 9), (9, "a".into(), 2)]);
}

#[test]
fn renumbering_follows_keep() {
    let mut g = LinkGraph::default();
    g.add(2, "a", 4).unwrap();
    let r = g.renumbered(&[0, 2, 4]);
    assert_eq!(r.all(), vec![(1, "a".into(), 2)]);
    assert!(r.pending().is_empty());
}

fn hit(id: u64, score: f32) -> SearchResult {
    SearchResult::new(id, score, None)
}

#[test]
fn expansion_is_breadth_first_and_bounded() {
    let mut g = LinkGraph::default();
    // chain 0 -> 1 -> 2 -> 3 -> 4, and 10 -> 1 (another type)
    for i in 0..4 {
        g.add(i, "next", i + 1).unwrap();
    }
    g.add(10, "other", 1).unwrap();
    let hits = [hit(0, 0.9)];
    let got = expand(&g, &hits, &Expansion::new("next", 2));
    assert_eq!(got, vec![(0, 0.9, 0), (1, 0.9, 1), (2, 0.9, 2)]);
    let got = expand(&g, &hits, &Expansion::new("next", 3).with_max_results(1));
    assert_eq!(got.len(), 2);
    // A record reached at two hops appears once, at the smaller one, with
    // the better parent's score
    g.add(5, "next", 2).unwrap();
    let got = expand(&g, &[hit(5, 0.8), hit(0, 0.5)], &Expansion::new("next", 3));
    assert_eq!(
        got,
        vec![
            (5, 0.8, 0),
            (0, 0.5, 0),
            (2, 0.8, 1),
            (1, 0.5, 1),
            (3, 0.8, 2),
            (4, 0.8, 3)
        ]
    );
}

#[test]
fn expansion_parameters_are_checked() {
    assert!(Expansion::new("a", 0).validate().is_err());
    assert!(Expansion::new("a", 4).validate().is_err());
    assert!(Expansion::new("A", 1).validate().is_err());
    assert!(Expansion::new("a", 1)
        .with_max_results(0)
        .validate()
        .is_err());
    assert!(Expansion::new("a", 1)
        .with_max_results(1001)
        .validate()
        .is_err());
    assert!(Expansion::new("a", 3)
        .with_max_results(1000)
        .validate()
        .is_ok());
}
