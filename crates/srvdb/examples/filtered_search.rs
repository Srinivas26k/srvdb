//! Filtered search example: create a database, add vectors, and search with metadata filtering

use anyhow::Result;
use srvdb::types::DatabaseConfig;
use srvdb::{SrvDB, Vector, VectorEngine};

fn main() -> Result<()> {
    println!("SrvDB {} filtered search example\n", env!("CARGO_PKG_VERSION"));

    // Initialize database configuration
    // 1536 is common for OpenAI embeddings
    let dimension = 1536;
    let config = DatabaseConfig::new(dimension)?;

    println!("Initializing database at ./demo_filtered_db with {} dimensions...", dimension);

    // Create database with explicit config
    let mut db = SrvDB::new_with_config("./demo_filtered_db", config)?;
    println!("✓ Database initialized\n");

    // Create some example vectors
    println!("Creating and adding vectors...");

    // Vector wrapper is still required for the Rust API
    let vec1 = Vector::new(vec![0.5; dimension]);
    let vec2 = Vector::new(vec![-0.3; dimension]);
    let vec3 = Vector::new(vec![0.8; dimension]);
    let vec4 = Vector::new(vec![0.2; dimension]);
    let vec5 = Vector::new(vec![-0.7; dimension]);

    // Add vectors with metadata
    let id1 = db.add(&vec1, r#"{"title": "Rust Programming", "category": "tech", "language": "rust"}"#)?;
    let id2 = db.add(&vec2, r#"{"title": "Machine Learning", "category": "science", "language": "python"}"#)?;
    let id3 = db.add(&vec3, r#"{"title": "Web Development", "category": "tech", "language": "javascript"}"#)?;
    let id4 = db.add(&vec4, r#"{"title": "Data Analysis", "category": "science", "language": "python"}"#)?;
    let id5 = db.add(&vec5, r#"{"title": "Database Design", "category": "tech", "language": "sql"}"#)?;

    println!("✓ Added 5 vectors with IDs: {}, {}, {}, {}, {}\n", id1, id2, id3, id4, id5);

    // Persist to ensure all data is searchable
    db.persist()?;

    // Build an allow-list using ids_where with a filter condition
    println!("Building allow-list for 'tech' category...");
    let tech_ids = db.ids_where(r#"{"category": "tech"}"#)?;
    println!("✓ Found {} tech documents: {:?}\n", tech_ids.len(), tech_ids);

    // Search for similar vectors only within the tech category
    println!("Searching for vectors similar to vec1 within 'tech' category...");
    let query = Vector::new(vec![0.51; dimension]); // Slightly different from vec1
    let filtered_results = db.search_filtered(&query, 3, Some(&tech_ids))?;

    println!("✓ Found {} filtered results:\n", filtered_results.len());
    for (i, result) in filtered_results.iter().enumerate() {
        println!(
            "  {}. ID: {} | Score: {:.4} | Metadata: {}",
            i + 1,
            result.id,
            result.score,
            result.metadata.as_ref().unwrap_or(&"None".to_string())
        );
    }

    // Build another allow-list for Python language documents
    println!("\nBuilding allow-list for 'python' language...");
    let python_ids = db.ids_where(r#"{"language": "python"}"#)?;
    println!("✓ Found {} Python documents: {:?}\n", python_ids.len(), python_ids);

    // Search for similar vectors only within Python documents
    println!("Searching for vectors similar to vec2 within 'python' language...");
    let query2 = Vector::new(vec![-0.31; dimension]); // Slightly different from vec2
    let python_results = db.search_filtered(&query2, 3, Some(&python_ids))?;

    println!("✓ Found {} filtered results:\n", python_results.len());
    for (i, result) in python_results.iter().enumerate() {
        println!(
            "  {}. ID: {} | Score: {:.4} | Metadata: {}",
            i + 1,
            result.id,
            result.score,
            result.metadata.as_ref().unwrap_or(&"None".to_string())
        );
    }

    // Demonstrate searching with an empty allow-list (should return no results)
    println!("\nSearching with empty allow-list...");
    let empty_ids = Vec::new();
    let empty_results = db.search_filtered(&query, 3, Some(&empty_ids))?;
    println!("✓ Found {} results (expected 0): {}\n", empty_results.len(), empty_results.len());

    // Demonstrate searching without an allow-list (should search all documents)
    println!("Searching without allow-list (all documents)...");
    let all_results = db.search_filtered(&query, 3, None)?;
    println!("✓ Found {} results in entire database:\n", all_results.len());
    for (i, result) in all_results.iter().enumerate() {
        println!(
            "  {}. ID: {} | Score: {:.4} | Metadata: {}",
            i + 1,
            result.id,
            result.score,
            result.metadata.as_ref().unwrap_or(&"None".to_string())
        );
    }

    println!("🎉 Filtered search demo completed!");

    Ok(())
}
