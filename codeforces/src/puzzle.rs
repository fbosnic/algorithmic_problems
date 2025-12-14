// This is the solution to the codeforces problem https://codeforces.com/problemset/problem/2111/F

use std::io::{BufRead, Read, stdin};
use std::collections::HashSet;


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Fraction {
    num: i32,
    den: i32,
}


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Coordinate {
    x: i32,
    y: i32,
}


#[derive(Debug, Clone)]
struct Solution {
    num_steps: usize;
    blocks: Vec<Coordinate>,
}


fn read_input<T: Read + BufRead>(
    input_stream: &mut T
) -> Vec<Fraction> {
    let mut input = String::new();
    input_stream.read_line(&mut input).unwrap();
    let n = input.trim().parse::<usize>().unwrap();
    let mut test_cases = Vec::with_capacity(n);
    for _ in 0..n {
        input = String::new();
        input_stream.read_line(&mut input).unwrap();
        let parts = input.trim().split_whitespace()
            .map(|x| x.parse::<i32>().unwrap()).collect::<Vec<i32>>();
        assert_eq!(parts.len(), 2);
        test_cases.push(Fraction { num: parts[0], den: parts[1] });
    }
    return test_cases;
}


fn solve(frac: &mut Fraction) -> Solution {
    if frac.num % 2 != 0 {
        frac.num *= 2;
        frac.den *= 2;
    }
    if 2 * (frac.den + 1) < frac.num {
        return Solution { num_steps: -1, blocks: Vec::new() };
    }
    let mut a = (frac.num - 2) / 4;
    let mut b = (frac.num - 2) - a;
    let k = frac.den / a * b + match frac.den % a {0 => 0, _ => 1};  // ceil operation
    frac.num *= k;
    frac.den *= k;
    a *= k;
    b *= k;

    let mut blocks = Vec::new();
    for idx in 0..frac.den {
        blocks.push(Coordinate { x:idx / a , y: idx % a });
    }
    return Solution { num_steps: blocks.len(), blocks: blocks };
}


fn main() {
    let test_cases = read_input(&mut stdin().lock());
    for frac in test_cases {
        let solution = solve(frac);
        println!("{}", solution.num_steps);
        for block in solution.blocks {
            println!("{} {}", block.x, block.y);
        }
    }
}


fn validate_solution(frac: Fraction, solution: Solution) {
    let den = solution.blocks.len();
    let mut num = 0;
    let placed: HashSet<Coordinate> = HashSet::new();
    for block in solution.blocks {
        num += 4;
        for x_dis in [-1, 1] {
            for y_dis in [-1, 1] {
                if placed.contains(&Coordinate { x: block.x + x_dis, y: block.y + y_dis }) {
                    sides -= 2;
                }
            }
        }

        if placed.contans(Coordinate { x: block.x, y: block.y }) {
            sides -= 1;
        }
        assert!(!placed.contains(&block));
        placed.insert(block);
        num += 1;
    }

    assert_eq!(solution.num_steps, solution.blocks.len());
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_input() {
        let input = "2\n1 2\n3 4\n";
        let mut input_stream = input.as_bytes();
        let test_cases = read_input(&mut input_stream);
        assert_eq!(test_cases.len(), 2);
        assert_eq!(test_cases[0], Fraction { num: 1, den: 2 });
        assert_eq!(test_cases[1], Fraction { num: 3, den: 4 });
    }
    #[test]
    fn test_1() {
        let frac = Fraction { num: 9, den: 2 };
        let solution = solve(frac);
        assert_eq!(solution.num_steps, -1);
    }

    fn test_2() {
        let frac: Fraction = Fraction { num: 23, den: 17 };
        let solution = solve(frac);
        assert_eq!(solution.num_steps, 13);
        assert_eq!(solution.blocks.len(), 13);
    }
}
