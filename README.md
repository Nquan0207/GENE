# **Genetic Algorithm for Solving the Traveling Salesman Problem (TSP)**  

## **Overview**  
This project implements a **Genetic Algorithm (GA)** to solve the **Traveling Salesman Problem (TSP)** efficiently. The TSP is a well-known combinatorial optimization problem that requires finding the shortest possible route that visits each city once and returns to the starting point.  

## **Key Features**  
- Implements core genetic operations: **Selection, Crossover, Mutation, and Fitness Evaluation**.  
- Compares different approaches: **Brute-force vs. Genetic Algorithm**.  
- Provides a convergence analysis to visualize performance.  
- Modularized Python scripts for easy experimentation and modification.  

## **Project Structure**  
```
📂 Genetic-TSP
│── .idea/                  # Project configuration files
│── __pycache__/            # Compiled Python files
│── TSP.py                  # Main script
│── compare_brut.py         # Brute-force comparison
│── convergence.py          # Convergence analysis
│── crossover.py            # Crossover operations
│── individual.py           # Individual representation
│── mutate.py               # Mutation operations
│── report.sty              # LaTeX report template
│── run.py                  # Execution script
│── selection.py            # Selection methods
│── test_case.py            # Sample test cases
│── value.py                # Fitness function
│── README.md               # Project documentation
```

## **Genetic Algorithm Process**  
1. **Initialization**: Generate an initial population of random solutions.  
2. **Selection**: Choose the best individuals based on fitness.  
3. **Crossover**: Combine parents to produce new offspring.  
4. **Mutation**: Introduce small changes to maintain diversity.  
5. **Evaluation**: Calculate fitness for new generations.  
6. **Termination**: Stop when the solution converges or reaches a set number of generations.  

## **Installation & Usage**  
**Prerequisites**:  
- Python 3.x  
- NumPy, Matplotlib (for analysis and visualization)  

**To run the algorithm:**  
```bash
git clone https://github.com/Nquan0207/Genetic-TSP
cd Genetic-TSP
python run.py
```

## **Results & Analysis**  
- The GA optimally finds a near-shortest route faster than brute-force approaches.  
- The **convergence.py** script visualizes how the best solution evolves over generations.  

## **References**  
- Goldberg, D. E. (1989). *Genetic Algorithms in Search, Optimization & Machine Learning*.  
- Holland, J. H. (1975). *Adaptation in Natural and Artificial Systems*.  

---
