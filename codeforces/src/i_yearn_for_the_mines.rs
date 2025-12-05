// Solution to the codeforces problem https://codeforces.com/problemset/problem/2133/E

use std::io::{stdin, Read, BufRead};
use std::collections::HashSet;


struct Node {
    neighbours: HashSet<usize>,
}

struct Graph {
    nodes: Vec<Node>,
}

impl Graph {
    pub fn create_disconnected_graph(n: usize) -> Self {
        let mut nodes = Vec::with_capacity(n);
        for _ in 0..n {
            nodes.push(Node { neighbours: HashSet::new() });
        }
        return Graph { nodes };
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

    pub fn destroy_edges_connected_to_node(&mut self, u: usize) {
        for &v in self.nodes[u].neighbours.iter() {
            self.nodes[v].neighbours.remove(&u);
        }
        self.nodes[u].neighbours.clear();
    }
}


struct SolvedCase {
    steps: Vec<Action>,
}


enum ActionType {
    INSPECT,
    DESTROY,
}


struct Action {
    action_type: ActionType,
    node_idx: usize,
}

impl Action {
    pub fn to_string(&self) -> String {
        match self.action_type {
            ActionType::INSPECT => format!("1 {}", self.node_idx + 1),
            ActionType::DESTROY => format!("2 {}", self.node_idx + 1),
        }
    }
}


fn read_input<T: Read + BufRead>(input_stream: &mut T) -> Vec<Graph> {
    let mut line = String::new();
    input_stream.read_line(&mut line).unwrap();
    let num_test_cases = line.trim().parse::<usize>().unwrap();
    let mut test_cases: Vec<Graph> = Vec::with_capacity(num_test_cases);
    for _test_idx in 0..num_test_cases {
        line = String::new();
        input_stream.read_line(&mut line).unwrap();
        let num_nodes = line.trim().parse::<usize>().unwrap();
        let mut graph = Graph::create_disconnected_graph(num_nodes);
        for _graph_idx in 0..num_nodes-1 {
            line = String::new();
            input_stream.read_line(&mut line).unwrap();
            let parts = line.trim().split_whitespace()
                .map(|x| x.parse::<usize>().unwrap()).collect::<Vec<usize>>();
            assert_eq!(parts.len(), 2);
            let a = parts[0] - 1;
            let b = parts[1] - 1;
            graph.add_edge(a, b);
        }
        test_cases.push(graph);
    }
    return test_cases;
}


fn solve_test_case(test_case: Graph) -> Vec<SolvedCase>{

}


fn main() {
    let test_cases = read_input(&mut stdin().lock());
    for tc in test_cases {
        println!("{}", solve_test_case(tc));
    }
}
