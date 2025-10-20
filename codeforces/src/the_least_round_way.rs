pub struct SquaerMatrix<T: Default + Copy> {
    _vector: Vec<T>,
    size: usize,
}

impl<T: Default + Copy> SquaerMatrix<T> {
    pub fn new(size: usize) -> Self {
        assert!(size <= 1000);
        Self { _vector: vec![T::default(); size*size], size: size }
    }

    pub fn at(&self, i: usize, j: usize) -> T {
        return self._vector[i * self.size + j].clone();
    }

    pub fn set(&mut self, i: usize, j: usize, value: T) {
        self._vector[i * self.size + j] = value;
    }

    pub fn apply(&self, func: fn(&T) -> T) -> Self {
        let mut new_matrix = SquaerMatrix::<T>::new(self.size);
        for i in 0..self.size {
            for j in 0..self.size {
                new_matrix.set(i, j, func(&self.at(i, j)));
            }
        }
        return new_matrix;
    }
}


fn read_input() -> SquaerMatrix<i64> {
    let mut size_line = String::new();
    std::io::stdin().read_line(&mut size_line).unwrap();
    let n = size_line.trim().parse::<usize>().unwrap();
    let mut matrix = SquaerMatrix::<i64>::new(n as usize);
    for row in 0..n {
        let mut input = String::new();
        std::io::stdin().read_line(&mut input).unwrap();
        let numbers: Vec<&str> = input.trim().split_whitespace().collect();
        assert!(numbers.len() == n as usize);
        for idx in 0..numbers.len(){
            let x = numbers[idx].parse::<i64>().unwrap();
            if x < 0 {
                println!("Matrix input negative - found {x}");
                panic!("Input must be positive");
            }
            matrix.set(row, idx, x);
        }
    }
    return matrix
}


fn count_power_of_factor(n: i64, factor: i64) -> i64 {
    let mut n = n;
    let mut count = 0;
    if n == 0 {
        return 1;
    }
    while n % factor == 0 {
        count += 1;
        n /= factor;
    }
    return count;
}


fn compute_minimal_weight_path(weight_matrix: &SquaerMatrix<i64>) -> (i64, String) {
    let n = weight_matrix.size;
    let mut dynamic_path_weight = SquaerMatrix::<i64>::new(n);
    let mut dynamic_previous_cell: SquaerMatrix<char> = SquaerMatrix::<char>::new(n);
    for row in 0..dynamic_path_weight.size {
        for col in 0..dynamic_path_weight.size {
            let weight = weight_matrix.at(row, col);
            if row == 0 && col == 0 {
                dynamic_path_weight.set(row, col, weight);
                dynamic_previous_cell.set(row, col, 'N');
            } else if row == 0 {
                dynamic_path_weight.set(
                    row,
                    col,
                    dynamic_path_weight.at(row, col - 1) + weight
                );
                dynamic_previous_cell.set(row, col, 'L');
            } else if col == 0 {
                dynamic_path_weight.set(
                    row,
                    col,
                    dynamic_path_weight.at(row - 1, col) + weight
                );
                dynamic_previous_cell.set(row, col, 'U');
            } else {
                if  dynamic_path_weight.at(row, col - 1) < dynamic_path_weight.at(row - 1, col) {
                    dynamic_previous_cell.set(row, col, 'L');
                    dynamic_path_weight.set(row, col, dynamic_path_weight.at(row, col - 1) + weight);
                } else {
                    dynamic_previous_cell.set(row, col, 'U');
                    dynamic_path_weight.set(row, col, dynamic_path_weight.at(row - 1, col) + weight);
                }
            }
        }
    }
    let mut path = String::new();
    let (mut x, mut y) = (n - 1, n - 1);
    while (x, y) != (0, 0) {
        match dynamic_previous_cell.at(x, y) {
            'L' => {
                path.push('R');
                y -= 1;
            }
            'U' => {
                path.push('D');
                x -= 1;
            }
            _ => panic!("Invalid content of cell"),
        }
    }
    path = path.chars().rev().collect();
    return (dynamic_path_weight.at(n - 1, n - 1), path)
}



fn find_least_round_way(matrix: SquaerMatrix<i64>) -> (i64, String) {
    let matrix_2_factors = matrix.apply(|&x| count_power_of_factor(x, 2));
    let matrix_5_factors = matrix.apply(|&x| count_power_of_factor(x, 5));

    let (min_weight_2, path_2) = compute_minimal_weight_path(&matrix_2_factors);
    let (min_weight_5, path_5) = compute_minimal_weight_path(&matrix_5_factors);
    let (mut min_weight, mut min_path) = if min_weight_2 < min_weight_5 {
        (min_weight_2, path_2)
    } else {
        (min_weight_5, path_5)
    };

    let mut coords_of_zero : (usize, usize) = (matrix.size, matrix.size);
    'outer: for row in 0..matrix.size {
        for col in 0..matrix.size {
            if matrix.at(row, col) == 0 {
                coords_of_zero = (row, col);
                break 'outer;
            }
        }
    }
    if (min_weight > 1) && (coords_of_zero != (matrix.size, matrix.size)) {
        min_weight = 1;
        let mut path = String::new();
        for _ in 0..coords_of_zero.0 {
            path.push('D');
        }
        for _ in 0..coords_of_zero.1 {
            path.push('R');
        }
        for _ in coords_of_zero.0..(matrix.size - 1) {
            path.push('D');
        }
        for _ in coords_of_zero.1..(matrix.size - 1) {
            path.push('R');
        }
        min_path = path;
    }
    return (min_weight, min_path);
}

fn main() {
    let matrix = read_input();
    let (minimal_trailing_zeros, path) = find_least_round_way(matrix);
    println!("{}", minimal_trailing_zeros);
    println!("{}", path);
}


// Tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_power_of_factor() {
        assert_eq!(count_power_of_factor(200, 2), 3);
        assert_eq!(count_power_of_factor(50, 5), 2);
    }

    #[test]
    fn test_compute_minimal_weight_path() {
        let data = vec![
            vec![0, 1, 7],
            vec![2, 2, 3],
            vec![3, 8, 0],
        ];
        let mut matrix = SquaerMatrix::<i64>::new(3);
        for row in 0..3 {
            for col in 0..3 {
                matrix.set(row, col, data[row][col]);
            }
        }
        let (minimal_weight, path) = compute_minimal_weight_path(&matrix);
        assert_eq!(minimal_weight, 6);
        assert_eq!(path, "RDRD");
    }

    #[test]
    fn test_least_round_way() {
        let data = vec![
            vec![1, 2, 3],
            vec![4, 5, 6],
            vec![7, 8, 9],
        ];
        let mut matrix = SquaerMatrix::<i64>::new(3);
        for row in 0..3 {
            for col in 0..3 {
                matrix.set(row, col, data[row][col]);
            }
        }
        let (minimal_trailing_zeros, path) = find_least_round_way(matrix);
        assert_eq!(minimal_trailing_zeros, 0);
        assert_eq!(path, "RRDD");
    }

    #[test]
    fn test_least_round_way_with_zero() {
        let data = vec![
            vec![5, 2, 2, 5],
            vec![2, 0, 1, 1],
            vec![5, 2, 10, 10],
            vec![5, 2, 10, 10],
        ];
        let mut matrix = SquaerMatrix::<i64>::new(4);
        for row in 0..4 {
            for col in 0..4 {
                matrix.set(row, col, data[row][col]);
            }
        }
        let (minimal_trailing_zeros, path) = find_least_round_way(matrix);
        assert_eq!(minimal_trailing_zeros, 1);
        assert_eq!(path, "DRDDRR");
    }
}