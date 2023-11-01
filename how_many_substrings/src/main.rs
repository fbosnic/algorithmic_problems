use std::cmp::min;
use std::vec::Vec;
use std::ops::{Range, Add, AddAssign, Neg, Mul};

extern crate algorithms;
use algorithms::data_structures::{RBTree, FenwickTree};


fn lcp_construction(str_bytes: &[u8], suffix_array: &Vec<usize>) -> Vec<usize> {
    let n = str_bytes.len();
    let sa_lookup = inverse_permutation(&suffix_array);
    let mut prefix_array = vec![0; n];
    let mut prefix_counter: usize = 0;
    for suffix_start in 0..n {
        let sa_pos = sa_lookup[suffix_start];
        if sa_pos == n-1 {
            prefix_counter = 0;
            continue;
        }
        let sa_follower_start = suffix_array[sa_pos + 1];
        while str_bytes[suffix_start + prefix_counter] == str_bytes[sa_follower_start + prefix_counter] {
            prefix_counter += 1;
        }
        prefix_array[sa_pos] = prefix_counter;
        if prefix_counter > 0 {
            prefix_counter -= 1;
        }
    }
    return prefix_array
}


fn inverse_permutation(permutation: &Vec<usize>) -> Vec<usize> {
    let n = permutation.len();
    let mut inverse_permutation: Vec<usize> = vec![0; n];
    for i in 1..n {
        inverse_permutation[permutation[i]] = i;
    }
    return inverse_permutation;
}


struct AdvanceFenwickTree<T: Add + AddAssign + Copy + Default + Neg> {
    linear_fwt: FenwickTree<T>,
    const_fwt: FenwickTree<T>
}


impl<T: Add<Output = T> + AddAssign + Copy + Default + Neg<Output = T> + Mul<i32, Output = T>> AdvanceFenwickTree<T> {
    pub fn add(&self, start: usize, end: usize, value: T) {
        self.linear_fwt.add(start, value);
        self.linear_fwt.add(end, -value);
        self.const_fwt.add(start, -value * i32::try_from(start).unwrap());
        self.const_fwt.add(end, value * i32::try_from(end).unwrap());
    }

    pub fn prefix_sum(&self, end: usize) -> T {
        let lin_part = self.linear_fwt.prefix_sum(end);
        let const_part = self.const_fwt.prefix_sum(end);
        return lin_part + const_part;
    }

    pub fn range_sum(&self, start: usize, end:usize) -> T {
        return self.prefix_sum(end) + (- self.prefix_sum(start));
    }

    pub fn with_len(len: usize) -> Self {
        AdvanceFenwickTree {
            linear_fwt: FenwickTree::with_len(len),
            const_fwt: FenwickTree::with_len(len)
        }
    }
}


fn count_substrings(str: String, queries: Vec<Range<usize> >) -> Vec<i32> {
    let n = str.len();
    let suffix_array = algorithms::string::generate_suffix_array_manber_myers(&str);
    let sa_lookup = inverse_permutation(&suffix_array);
    let lcp = lcp_construction(str.as_bytes(), suffix_array);
    let min_lcp_segment_tree = algorithms::data_structures::SegmentTree::from_vec(&lcp, min);

    let mut suffix_sorter: RBTree<usize, i8> = RBTree::new();
    let fw_substr_count : FenwickTree<u32> = FenwickTree::with_len(n);
    let fw_lcp_count: FenwickTree<u32> = FenwickTree::with_len(n);

    for suffix in (0..n-1).rev() {
        let sa_pos = sa_lookup[suffix];
        suffix_sorter.insert(sa_pos, 0);
        let node = suffix_sorter.find_node(&sa_pos);
        if let Some(i) = node {
            println!("{}", i.key);
        }
    }
    println!("{lcp:?}");
    return vec![1, 2, 3]
}


fn main() {
    let str = String::from("value");
    let queries: Vec<Range<usize>> = vec![Range{start: 0, end: 1}, Range{start: 2, end: 3}];
    let results = count_substrings(str, queries);
}
