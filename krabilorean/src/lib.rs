#![cfg_attr(not(test), no_std)]
#![forbid(unsafe_code)]

#[cfg(test)]
mod tests {
    #[test]
    fn unit_tests_may_use_std() {
        let values = std::vec![1_u8, 2, 3];
        assert_eq!(values.len(), 3);
    }
}
