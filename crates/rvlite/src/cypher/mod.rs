//! Cypher query language parser and execution engine for WASM
//!
//! This module provides a WASM-compatible Cypher implementation including:
//! - Lexical analysis (tokenization)
//! - Syntax parsing (AST generation)
//! - In-memory property graph storage
//! - Query execution engine
//!
//! Supported operations:
//! - CREATE: Create nodes and relationships
//! - MATCH: Pattern matching
//! - WHERE: Filtering
//! - RETURN: Projection
//! - SET: Update properties
//! - DELETE/DETACH DELETE: Remove nodes and edges

pub mod ast;
pub mod executor;
pub mod graph_store;
pub mod lexer;
pub mod parser;

pub use ast::{Expression, Pattern, Query, Statement};
pub use executor::{ContextValue, ExecutionError, ExecutionResult, Executor};
pub use graph_store::{Edge, EdgeId, Node, NodeId, PropertyGraph, Value};
pub use lexer::{tokenize, Token, TokenKind};
pub use parser::{parse_cypher, ParseError};

use crate::storage::state::{EdgeState, GraphState, NodeState, PropertyValue};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use wasm_bindgen::prelude::*;

/// WASM-compatible Cypher engine
#[wasm_bindgen]
pub struct CypherEngine {
    graph: PropertyGraph,
}

#[wasm_bindgen]
impl CypherEngine {
    /// Create a new Cypher engine with empty graph
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            graph: PropertyGraph::new(),
        }
    }

    /// Execute a Cypher query and return JSON results
    pub fn execute(&mut self, query: &str) -> Result<JsValue, JsValue> {
        // Parse the query
        let ast =
            parse_cypher(query).map_err(|e| JsValue::from_str(&format!("Parse error: {}", e)))?;

        // Execute the query
        let mut executor = Executor::new(&mut self.graph);
        let result = executor
            .execute(&ast)
            .map_err(|e| JsValue::from_str(&format!("Execution error: {}", e)))?;

        // Convert to JS value
        serde_wasm_bindgen::to_value(&result)
            .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
    }

    /// Get graph statistics
    pub fn stats(&self) -> Result<JsValue, JsValue> {
        let stats = self.graph.stats();
        serde_wasm_bindgen::to_value(&stats)
            .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
    }

    /// Clear the graph
    pub fn clear(&mut self) {
        self.graph = PropertyGraph::new();
    }
}

impl CypherEngine {
    /// Export graph state for persistence
    pub fn export_state(&self) -> GraphState {
        let nodes: Vec<NodeState> = self
            .graph
            .all_nodes()
            .into_iter()
            .map(|n| NodeState {
                id: n.id.clone(),
                labels: n.labels.clone(),
                properties: n
                    .properties
                    .iter()
                    .map(|(k, v)| (k.clone(), value_to_property(v)))
                    .collect(),
            })
            .collect();

        let edges: Vec<EdgeState> = self
            .graph
            .all_edges()
            .into_iter()
            .map(|e| EdgeState {
                id: e.id.clone(),
                from: e.from.clone(),
                to: e.to.clone(),
                edge_type: e.edge_type.clone(),
                properties: e
                    .properties
                    .iter()
                    .map(|(k, v)| (k.clone(), value_to_property(v)))
                    .collect(),
            })
            .collect();

        let stats = self.graph.stats();

        GraphState {
            nodes,
            edges,
            next_node_id: stats.node_count,
            next_edge_id: stats.edge_count,
        }
    }

    /// Import state to restore the graph
    pub fn import_state(&mut self, state: &GraphState) -> Result<(), JsValue> {
        self.graph = PropertyGraph::new();

        // Import nodes first
        for node_state in &state.nodes {
            let mut node = Node::new(node_state.id.clone());
            for label in &node_state.labels {
                node = node.with_label(label.clone());
            }
            for (key, value) in &node_state.properties {
                node = node.with_property(key.clone(), property_to_value(value));
            }
            self.graph.add_node(node);
        }

        // Then import edges
        for edge_state in &state.edges {
            let mut edge = Edge::new(
                edge_state.id.clone(),
                edge_state.from.clone(),
                edge_state.to.clone(),
                edge_state.edge_type.clone(),
            );
            for (key, value) in &edge_state.properties {
                edge = edge.with_property(key.clone(), property_to_value(value));
            }
            if let Err(e) = self.graph.add_edge(edge) {
                return Err(JsValue::from_str(&format!("Failed to import edge: {}", e)));
            }
        }

        Ok(())
    }
}

impl Default for CypherEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert graph Value to serializable PropertyValue
fn value_to_property(v: &Value) -> PropertyValue {
    match v {
        Value::Null => PropertyValue::Null,
        Value::Boolean(b) => PropertyValue::Boolean(*b),
        Value::Integer(i) => PropertyValue::Integer(*i),
        Value::Float(f) => PropertyValue::Float(*f),
        Value::String(s) => PropertyValue::String(s.clone()),
        Value::List(list) => PropertyValue::List(list.iter().map(value_to_property).collect()),
        Value::Map(map) => PropertyValue::Map(
            map.iter()
                .map(|(k, v)| (k.clone(), value_to_property(v)))
                .collect(),
        ),
    }
}

/// Convert PropertyValue back to graph Value
fn property_to_value(p: &PropertyValue) -> Value {
    match p {
        PropertyValue::Null => Value::Null,
        PropertyValue::Boolean(b) => Value::Boolean(*b),
        PropertyValue::Integer(i) => Value::Integer(*i),
        PropertyValue::Float(f) => Value::Float(*f),
        PropertyValue::String(s) => Value::String(s.clone()),
        PropertyValue::List(list) => Value::List(list.iter().map(property_to_value).collect()),
        PropertyValue::Map(map) => Value::Map(
            map.iter()
                .map(|(k, v)| (k.clone(), property_to_value(v)))
                .collect(),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_node() {
        let mut engine = CypherEngine::new();
        let query = "CREATE (n:Person {name: 'Alice', age: 30})";

        let ast = parse_cypher(query).unwrap();
        let mut executor = Executor::new(&mut engine.graph);
        let result = executor.execute(&ast);

        assert!(result.is_ok());
        assert_eq!(engine.graph.stats().node_count, 1);
    }

    #[test]
    fn test_create_relationship() {
        let mut engine = CypherEngine::new();
        let query = "CREATE (a:Person {name: 'Alice'})-[r:KNOWS]->(b:Person {name: 'Bob'})";

        let ast = parse_cypher(query).unwrap();
        let mut executor = Executor::new(&mut engine.graph);
        let result = executor.execute(&ast);

        assert!(result.is_ok());
        let stats = engine.graph.stats();
        assert_eq!(stats.node_count, 2);
        assert_eq!(stats.edge_count, 1);
    }

    /// Issue #269: MATCH must return ALL matching rows, not just the last one.
    /// This was the critical bug — context.bind() overwrote previous bindings.
    #[test]
    fn test_match_returns_multiple_rows() {
        let mut engine = CypherEngine::new();

        // Create 3 Person nodes
        let create = "CREATE (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}), (c:Person {name: 'Charlie'})";
        let ast = parse_cypher(create).unwrap();
        let mut executor = Executor::new(&mut engine.graph);
        executor.execute(&ast).unwrap();
        assert_eq!(engine.graph.stats().node_count, 3);

        // MATCH all Person nodes — must return 3 rows
        let match_query = "MATCH (n:Person) RETURN n";
        let ast = parse_cypher(match_query).unwrap();
        let mut executor = Executor::new(&mut engine.graph);
        let result = executor.execute(&ast).unwrap();

        assert_eq!(
            result.rows.len(),
            3,
            "MATCH (n:Person) RETURN n should return 3 rows for 3 Person nodes, got {}",
            result.rows.len()
        );
        assert_eq!(result.columns, vec!["n"]);
    }

    /// Verify MATCH with property access returns correct values for each row.
    #[test]
    fn test_match_return_properties() {
        let mut engine = CypherEngine::new();

        let create = "CREATE (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'})";
        let ast = parse_cypher(create).unwrap();
        let mut executor = Executor::new(&mut engine.graph);
        executor.execute(&ast).unwrap();

        let match_query = "MATCH (n:Person) RETURN n.name";
        let ast = parse_cypher(match_query).unwrap();
        let mut executor = Executor::new(&mut engine.graph);
        let result = executor.execute(&ast).unwrap();

        assert_eq!(result.rows.len(), 2, "Should return 2 rows");

        // Collect returned names
        let mut names: Vec<String> = result
            .rows
            .iter()
            .filter_map(|row| {
                row.get("n.name").and_then(|v| {
                    if let ContextValue::Value(Value::String(s)) = v {
                        Some(s.clone())
                    } else {
                        None
                    }
                })
            })
            .collect();
        names.sort();
        assert_eq!(names, vec!["Alice", "Bob"]);
    }

    /// Verify MATCH with WHERE correctly filters results.
    #[test]
    fn test_match_where_filter() {
        let mut engine = CypherEngine::new();

        let create =
            "CREATE (a:Person {name: 'Alice', age: 30}), (b:Person {name: 'Bob', age: 25}), (c:Person {name: 'Charlie', age: 35})";
        let ast = parse_cypher(create).unwrap();
        let mut executor = Executor::new(&mut engine.graph);
        executor.execute(&ast).unwrap();

        // Match persons with age > 28
        let match_query = "MATCH (n:Person) WHERE n.age > 28 RETURN n.name";
        let ast = parse_cypher(match_query).unwrap();
        let mut executor = Executor::new(&mut engine.graph);
        let result = executor.execute(&ast).unwrap();

        assert_eq!(
            result.rows.len(),
            2,
            "Should return 2 rows (Alice=30 and Charlie=35), got {}",
            result.rows.len()
        );
    }

    /// Test with a single match — should still return exactly 1 row.
    #[test]
    fn test_match_single_result() {
        let mut engine = CypherEngine::new();

        let create = "CREATE (a:Person {name: 'Alice'})";
        let ast = parse_cypher(create).unwrap();
        let mut executor = Executor::new(&mut engine.graph);
        executor.execute(&ast).unwrap();

        let match_query = "MATCH (n:Person) RETURN n";
        let ast = parse_cypher(match_query).unwrap();
        let mut executor = Executor::new(&mut engine.graph);
        let result = executor.execute(&ast).unwrap();

        assert_eq!(result.rows.len(), 1, "Should return exactly 1 row");
    }

    /// Test with no matches — should return 0 rows.
    #[test]
    fn test_match_no_results() {
        let mut engine = CypherEngine::new();

        // Create a Person but match for Animal
        let create = "CREATE (a:Person {name: 'Alice'})";
        let ast = parse_cypher(create).unwrap();
        let mut executor = Executor::new(&mut engine.graph);
        executor.execute(&ast).unwrap();

        let match_query = "MATCH (n:Animal) RETURN n";
        let ast = parse_cypher(match_query).unwrap();
        let mut executor = Executor::new(&mut engine.graph);
        let result = executor.execute(&ast).unwrap();

        assert_eq!(result.rows.len(), 0, "Should return 0 rows for no matches");
    }

    /// Test MATCH with many nodes — stress test for the multi-row fix.
    #[test]
    fn test_match_many_nodes() {
        let mut engine = CypherEngine::new();

        // Create 100 nodes
        for i in 0..100 {
            let create = format!("CREATE (n:Item {{id: {}}})", i);
            let ast = parse_cypher(&create).unwrap();
            let mut executor = Executor::new(&mut engine.graph);
            executor.execute(&ast).unwrap();
        }
        assert_eq!(engine.graph.stats().node_count, 100);

        // MATCH all — must return 100 rows
        let match_query = "MATCH (n:Item) RETURN n";
        let ast = parse_cypher(match_query).unwrap();
        let mut executor = Executor::new(&mut engine.graph);
        let result = executor.execute(&ast).unwrap();

        assert_eq!(
            result.rows.len(),
            100,
            "MATCH should return all 100 nodes, got {}",
            result.rows.len()
        );
    }

    #[test]
    fn test_match_nodes_basic() {
        let mut engine = CypherEngine::new();

        // Create data
        let create = "CREATE (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'})";
        let ast = parse_cypher(create).unwrap();
        let mut executor = Executor::new(&mut engine.graph);
        executor.execute(&ast).unwrap();

        // Match nodes
        let match_query = "MATCH (n:Person) RETURN n";
        let ast = parse_cypher(match_query).unwrap();
        let mut executor = Executor::new(&mut engine.graph);
        let result = executor.execute(&ast);

        assert!(result.is_ok());
    }

    #[test]
    fn test_parser() {
        let queries = vec![
            "MATCH (n:Person) RETURN n",
            "CREATE (n:Person {name: 'Alice'})",
            "MATCH (a)-[r:KNOWS]->(b) RETURN a, r, b",
            "CREATE (a:Person)-[r:KNOWS]->(b:Person)",
        ];

        for query in queries {
            let result = parse_cypher(query);
            assert!(result.is_ok(), "Failed to parse: {}", query);
        }
    }
}
