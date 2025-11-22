use std::io::{stdin, Read, BufRead};
use std::collections::BinaryHeap;


fn read_input<T: Read + BufRead>(input_stream: &mut T) -> (usize, usize, Vec<i32>) {
    let mut line= String::new();
    input_stream.read_line(&mut line).unwrap();
    let _parts = line.trim().split_whitespace()
        .map(|x| x.parse::<usize>().unwrap()).collect::<Vec<usize>>();
    let n = _parts[0];
    let k = _parts[1];

    let mut line = String::new();
    input_stream.read_line(&mut line).unwrap();
    let array = line.trim().split_whitespace()
        .map(|x| x.parse::<i32>().unwrap()).collect::<Vec<i32>>();
    assert_eq!(array.len(), n);
    return (n, k, array);
}


struct CarrotCutImprovement {
    carrot_idx: usize,
    time_decrease: i32,
}


impl PartialEq for CarrotCutImprovement {
    fn eq(&self, other: &Self) -> bool {
        self.time_decrease == other.time_decrease
    }
}


impl Eq for CarrotCutImprovement {}


impl PartialOrd for CarrotCutImprovement {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        return Some(self.time_decrease.cmp(&other.time_decrease));
    }
}


impl Ord for CarrotCutImprovement {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.time_decrease.cmp(&other.time_decrease)
    }
}


fn compute_carrot_eat_time(carrot_size: i32, num_cuts: i32) -> i32 {
    let reminder = carrot_size % (num_cuts + 1);
    return (carrot_size - reminder).pow(2) / (num_cuts + 1)
        + 2 * (carrot_size - reminder) * reminder / (num_cuts + 1) + reminder;
}


fn compute_decrease_for_cut(carrot_size: i32, num_initial_cuts: i32) -> i32 {
    return compute_carrot_eat_time(carrot_size, num_initial_cuts)
        - compute_carrot_eat_time(carrot_size, num_initial_cuts + 1);
}


fn compute_eat_time_for_rabbits(n: usize, k: usize, carrots: Vec<i32>) -> i32 {
    let mut cuts: Vec<i32> = vec![0; n];
    let mut heap: BinaryHeap<CarrotCutImprovement> = BinaryHeap::with_capacity(n);
    for idx in 0..n {
        heap.push(CarrotCutImprovement {
            carrot_idx: idx,
            time_decrease: compute_decrease_for_cut(carrots[idx], cuts[idx]),
        });
    }
    for _ in 0..(k - n) {
        let max_element = heap.pop().unwrap();
        let carrot_idx = max_element.carrot_idx;
        cuts[max_element.carrot_idx] += 1;
        heap.push(
            CarrotCutImprovement {
                carrot_idx: max_element.carrot_idx,
                time_decrease: compute_decrease_for_cut(carrots[carrot_idx], cuts[carrot_idx]),
            }
        );
    }

    let mut total_eat_time = 0;
    for idx in 0..n {
        total_eat_time += compute_carrot_eat_time(carrots[idx], cuts[idx]);
    }
    return total_eat_time;
}


fn solve<T: Read + BufRead>(input_stream: &mut T) -> i32{
    let (n, k, carrots) = read_input(input_stream);
    return compute_eat_time_for_rabbits(n, k, carrots);
}


fn main() {
    let result = solve(& mut stdin().lock());
    println!("{}", result);
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_carrot_eat_time() {
        assert_eq!(compute_carrot_eat_time(5, 0), 25);
        assert_eq!(compute_carrot_eat_time(5, 1), 13);
        assert_eq!(compute_carrot_eat_time(5, 2), 9);
        assert_eq!(compute_carrot_eat_time(5, 3), 7);
    }

    #[test]
    fn example1() {
        let input = "3 6\n 5 3 1 ".as_bytes();
        let mut input_stream = std::io::BufReader::new(input);
        let solution = solve(& mut input_stream);
        assert_eq!(solution, 15);
    }

    #[test]
    fn example2() {
        let input = "1 4\n 19 ".as_bytes();
        let mut input_stream = std::io::BufReader::new(input);
        let solution = solve(& mut input_stream);
        assert_eq!(solution, 91);
    }
}
