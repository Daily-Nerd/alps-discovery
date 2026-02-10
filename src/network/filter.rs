// ALPS Discovery — Metadata Filtering
//
// Post-scoring filter conditions for metadata-based result filtering.

use std::collections::HashMap;

/// A filter condition for metadata-based result filtering.
#[derive(Debug, Clone)]
pub enum FilterValue {
    /// Exact string match.
    Exact(String),
    /// Substring containment check.
    Contains(String),
    /// Value must be one of the listed options.
    OneOf(Vec<String>),
    /// Numeric value must be less than the threshold.
    LessThan(f64),
    /// Numeric value must be greater than the threshold.
    GreaterThan(f64),
}

impl FilterValue {
    /// Check if a metadata value matches this filter condition.
    pub fn matches(&self, value: &str) -> bool {
        match self {
            FilterValue::Exact(expected) => value == expected,
            FilterValue::Contains(substring) => value.contains(substring.as_str()),
            FilterValue::OneOf(options) => options.iter().any(|o| o == value),
            FilterValue::LessThan(threshold) => value.parse::<f64>().is_ok_and(|v| v < *threshold),
            FilterValue::GreaterThan(threshold) => {
                value.parse::<f64>().is_ok_and(|v| v > *threshold)
            }
        }
    }
}

/// Metadata filters applied to discovery results.
pub type Filters = HashMap<String, FilterValue>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_match() {
        let f = FilterValue::Exact("mcp".to_string());
        assert!(f.matches("mcp"));
        assert!(!f.matches("grpc"));
    }

    #[test]
    fn contains_match() {
        let f = FilterValue::Contains("legal".to_string());
        assert!(f.matches("legal-v2"));
        assert!(!f.matches("medical"));
    }

    #[test]
    fn one_of_match() {
        let f = FilterValue::OneOf(vec!["a".to_string(), "b".to_string()]);
        assert!(f.matches("a"));
        assert!(f.matches("b"));
        assert!(!f.matches("c"));
    }

    #[test]
    fn less_than_match() {
        let f = FilterValue::LessThan(10.0);
        assert!(f.matches("5"));
        assert!(!f.matches("15"));
        assert!(!f.matches("not_a_number"));
    }

    #[test]
    fn greater_than_match() {
        let f = FilterValue::GreaterThan(1.0);
        assert!(f.matches("2.5"));
        assert!(!f.matches("0.5"));
    }
}
