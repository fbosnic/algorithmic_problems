use std::ops::Deref;


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

    pub fn apply(&self, func: fn(&T) -> T) {
        let mut new_matrix = SquaerMatrix::<T>::new(self.size);
        for i in 0..self.size {
            for j in 0..self.size {
                new_matrix.set(i, j, func(&self.at(i, j)));
            }
        }
    }
}


fn read_input() -> SquaerMatrix<i32> {
    let mut size_line = String::new();
    std::io::stdin().read_line(&mut size_line).unwrap();
    let n = size_line.trim().parse::<usize>().unwrap();
    let mut matrix = SquaerMatrix::<i32>::new(n as usize);
    for row in 0..n {
        let mut input = String::new();
        std::io::stdin().read_line(&mut input).unwrap();
        let numbers: Vec<&str> = input.trim().split_whitespace().collect();
        assert!(numbers.len() == n as usize);
        for idx in 0..numbers.len(){
            matrix.set(row, idx, numbers[idx].parse::<i32>().unwrap());
        }
    }
    return matrix
}


fn count_power_of_factor(n: i32, factor: i32) -> i32 {
    let mut n = n;
    let mut count = 0;
    while n % factor == 0 {
        count += 1;
        n /= factor;
    }
    return count;
}


fn compute_minimal_weight_path(weight_matrix: &SquaerMatrix<i32>) -> SquaerMatrix<i32> {
    let mut dynamic_path_weight = SquaerMatrix::<i32>::new(weight_matrix.size);
    let mut dynamic_previous_cell: SquaerMatrix<char> = SquaerMatrix::<char>::new(weight_matrix.size);
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
    return dynamic_path_weight
}



fn main() {
    let matrix = read_input();
    let matrix_2_factors = matrix.apply(|&x| count_power_of_factor(x, 2));
    let matrix_5_factors = matrix.apply(|&x| count_power_of_factor(x, 5));

    return std::cmp::min(v1, v2)
}