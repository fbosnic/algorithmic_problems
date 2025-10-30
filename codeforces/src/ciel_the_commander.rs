use std::io::{stdin, Read, BufRead};
use std::collections::{HashSet, VecDeque};

struct Node {
    neighbours: HashSet<usize>,
}

struct TreeGraph {
    nodes: Vec<Node>,
}

impl TreeGraph {
    pub fn create_disconnected_graph(n: usize) -> Self {
        let mut nodes = Vec::with_capacity(n);
        for _ in 0..n {
            nodes.push(Node { neighbours: HashSet::new() });
        }
        return TreeGraph { nodes };
    }

    pub fn add_edge(&mut self, u: usize, v: usize) {
        self.nodes[u].neighbours.insert(v);
        self.nodes[v].neighbours.insert(u);
    }

    pub fn neighbours(&self, u: usize) -> &HashSet<usize> {
        return &self.nodes[u].neighbours;
    }

    pub fn leaves(&self) -> Vec<usize> {
        let mut leaves = Vec::new();
        for (i, node) in self.nodes.iter().enumerate() {
            if node.neighbours.len() == 1 {
                leaves.push(i);
            }
        }
        return leaves;
    }
}


fn read_input<T: Read + BufRead>(input_stream: &mut T) -> TreeGraph {
    let mut input = String::new();
    input_stream.read_line(&mut input).unwrap();
    let n = input.trim().parse::<usize>().unwrap();
    let mut graph = TreeGraph::create_disconnected_graph(n);

    for _ in 0..n-1 {
        let mut input = String::new();
        input_stream.read_line(&mut input).unwrap();
        let v: Vec<usize> = input.trim().split_whitespace()
            .map(|x| x.parse::<usize>().unwrap()).collect::<Vec<usize>>();
        assert_eq!(v.len(), 2, "Expected 2 numbers, found {}", v.len());
        let x = v[0] - 1;
        let y = v[1] - 1;
        graph.add_edge(x, y);
    }
    return graph;
}


fn solve<T: Read + BufRead>(input_stream: &mut T) -> String {
    let graph = read_input(input_stream);
    let mut assigments: Vec<i32> = vec![0; graph.nodes.len()];
    let mut node_queue: VecDeque<usize> = VecDeque::new();
    for leaf in graph.leaves() {
        assigments[leaf] = 1;
        node_queue.push_back(leaf);
    }
    while node_queue.len() > 0 {
        let node_idx = node_queue.pop_front().unwrap();
        for &neighbour in graph.neighbours(node_idx) {
            if assigments[neighbour] != 0 {
                continue;
            }
            assigments[neighbour] = assigments[node_idx] + 1;
            node_queue.push_back(neighbour);
        }
    }
    let max_rank: i32 = *assigments.iter().max().unwrap();
    if max_rank > 26 {
        return "Impossible!".to_string();
    }
    dbg!(&assigments);
    let mut result = String::new();
    for a in assigments {
        result.push((b'A' + (a - 1) as u8) as char);
        result.push(' ');
    }
    result.pop();
    return result;
}

fn main() {
    let result: String = solve(& mut stdin().lock());
    println!("{result}");
}


#[cfg(test)]

mod tests {

    #[test]
    fn simple_example() {
        let input = "4\n1 2\n1 3\n1 4   \n".as_bytes();
        let mut input_stream = std::io::BufReader::new(input);
        let result: String = super::solve(& mut input_stream);
        assert_eq!(result, "A B B A");
    }
}
