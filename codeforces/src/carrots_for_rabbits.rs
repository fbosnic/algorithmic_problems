use std::io::{stdin, Read, BufRead};


fn read_input<T: Read + BufRead>(input_stream: &mut T) -> (usize, usize, Vec<i32̣>) {
    let (n, k, carrots) = read_input(input_stream);
    let mut line= String::new();
    input_stream.read_line(line).unwrap();
    let _parts = line.trim().split_whitespace()
        .map(|x| x.parse::<usize>().unwrap()).collect::<Vec<usize>>();
    let n = _parts[0];
    let k = _parts[1];

    input_stream.read_line(line).unwrap();
    let array = line.trim().split_whitespace()
        .map(|x| x.parse::<i32>().unwrap()).collect::<Vec<i32>>();
    assert_eq!(array.len(), n);
    return (n, k, array);
}


fn compute_eat_time_for_rabbits(n: usize, k: usize, mut carrots: Vec<i32>) -> i32 {
    let mut total_time = 0;
    let sorted = carrots.sort();
    return total_time;
}


fn solve<T: Read + BufRead>(input_stream: &mut T) -> i32 {
    let (n, k, carrots) = read_input(input_stream);
    compute_eat_time_for_rabbits(n, k, carrots);
}


fn main() {
    solve(& mut stdin().lock());
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn example1() {
        let input = "3 6\n 5 3 1 ".as_bytes();
        let mut input_stream = std::io::BufReader::new(input);
        let solution = solve(& mut input_stream);
        assert_eq!(solution, 15);
    }
}
