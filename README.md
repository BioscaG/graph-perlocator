# 🔍 Graph Percolator

Graph Percolator is a tool for studying **graph percolation**. It allows simulations on different types of graphs and analyzes connectivity properties as a function of probability.

## 📌 Features
- Supports different **graph types**: `g2d`, `rgg`, `g3d`, `ccg`
- **Customizable parameters**: Graph size, number of tests, probability values.
- **Edge vs. Vertex percolation**: Choose between edge or vertex percolation.
- **Visualization tools**: Generates CSV data and graphical representations.

---

## 🚀 Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/graph-perlocator.git
cd graph-perlocator
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Prepare the data
Extract any required datasets:
```bash
tar -xvf data.tar.gz  # If applicable
```

---

## ⚙️ Usage

### Run a percolation simulation
Execute the main script:
```bash
python3 graphPerlocator_beta.py
```

### Generate plots
To visualize the results:
```bash
python3 plotter.py
```

### Configuration
Modify `config.ini` to set up the experiment:
```ini
[config]
graph_type = rgg
n = 10, 20, 40, 60
k = 100
q = 100
perl_vertex = 0  # 0 for edge percolation, 1 for vertex percolation
```

---

## 📊 Output
- **CSV Files**: Data is stored in `data/ccg/`.
- **Graphical Results**: Images are saved as `.png` in the same folder.

---

## 📜 License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## 🤝 Contributions
Contributions are welcome! If you'd like to improve the tool, open a pull request.

---

## 📬 Contact
For any inquiries, contact:
📧 guido.biosca0@gmail.com

