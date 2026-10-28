import pandas as pd

class Matrix():
    def __init__(self, dense_matrix):
        self.matrix = dense_matrix
        self.rows = len(dense_matrix)
        self.cols = len(dense_matrix[0]) if self.rows > 0 else 0 

    # COO
    def to_coo(self):
        result = [["Value", "Row", "Col"]]
        for i in range(self.rows):
            for j in range(self.cols):
                if self.matrix[i][j] != 0:
                    result.append([self.matrix[i][j], i, j])
        return result
    
    # CSR
    def to_csr(self):
        data = [["Value", "Col"]]
        row_index = [0]
        non_zeros = 0

        for i in range(self.rows):
            for j in range(self.cols):
                if self.matrix[i][j] != 0:
                    data.append([self.matrix[i][j], j])
                    non_zeros += 1
            row_index.append(non_zeros)
            
        return data, row_index

    # ELLPACK
    def to_ellpack(self):
        max_nnz = 0
        for i in range(self.rows):
            nnz = sum(1 for x in self.matrix[i] if x != 0)
            max_nnz = max(max_nnz, nnz)
            
        ell_values = [[0] * max_nnz for _ in range(self.rows)]
        ell_cols = [[0] * max_nnz for _ in range(self.rows)]
        
        for i in range(self.rows):
            col_idx = 0
            for j in range(self.cols):
                if self.matrix[i][j] != 0:
                    ell_values[i][col_idx] = self.matrix[i][j]
                    ell_cols[i][col_idx] = j
                    col_idx += 1
                    
        return ell_values, ell_cols

    # Очистка данных
    def clean(self):
        active_cols = []
        for j in range(self.cols):
            if any(self.matrix[i][j] != 0 for i in range(self.rows)):
                active_cols.append(j)

        temp_matrix = []
        for i in range(self.rows):
            new_row = [self.matrix[i][j] for j in active_cols]
            temp_matrix.append(new_row)

        clean_matrix = []
        for row in temp_matrix:
            valid_ratings = [x for x in row if x != 0]

            if len(valid_ratings) > 0:
                mid = sum(valid_ratings) / len(valid_ratings)
                
                if mid >= 4:
                    clean_matrix.append(row)

        return Matrix(clean_matrix)


dense_matrix_1 = pd.read_csv('./matrix1.csv', header=None).values.tolist()
dense_matrix_2 = pd.read_csv('./matrix2.csv', header=None).values.tolist()

matrix_1 = Matrix(dense_matrix_1)

print("Исходная матрица №1:")
for row in matrix_1.matrix:
    print(row)

clean_matrix_1 = matrix_1.clean()

print("\nОчищенная матрица №1:")
for row in clean_matrix_1.matrix:
    print(row)

print("\nCOO format (для очищенной):")
for row in clean_matrix_1.to_coo():
    print(row)

print("\nCRS format (для очищенной):")
csr_matrix, row_index = clean_matrix_1.to_csr() 
for row in csr_matrix:
    print(row)
print(f"RowIndex: {row_index}")

print("\nEllpack format (для очищенной):")
ell_values, ell_cols = clean_matrix_1.to_ellpack() 
print("Value:")
for row in ell_values:
    print(row)
print("Column:")
for row in ell_cols:
    print(row)


matrix_2 = Matrix(dense_matrix_2)

print("\n\nИсходная матрица №2:")
for row in matrix_2.matrix:
    print(row)

clean_matrix_2 = matrix_2.clean()

print("\nОчищенная матрица №2:")
for row in clean_matrix_2.matrix:
    print(row)

print("\nCOO format (для очищенной):")
for row in clean_matrix_2.to_coo():
    print(row)

print("\nCRS format (для очищенной):")
csr_matrix, row_index = clean_matrix_2.to_csr()
for row in csr_matrix:
    print(row)
print(f"RowIndex: {row_index}")

print("\nEllpack format (для очищенной):")
ell_values, ell_cols = clean_matrix_2.to_ellpack()
print("Value:")
for row in ell_values:
    print(row)
print("Column:")
for row in ell_cols:
    print(row)