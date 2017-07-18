require 'matrix'

Math_E = Math.exp(1)

class Matrix
  def []=(i, j, x)
    @rows[i][j] = x
  end
end

def sigmoid(x, deriv=false)
  if deriv == true
    return x.map { |i| i * (1 - i)}
  else
    return x.map { |i| 1/(1 + Math.exp(-i)) }
  end
end

def matrix_term_multiply(a, b)
  t = Matrix.zero(a.row_count, a.column_count)
  a.each_with_index do |e, row, col|
    t[row, col] = e * b[row, col]
  end
  return t
end

x = Matrix[  [0,0,1],
  [0,1,1],
  [1,0,1],
  [1,1,1] 
]

y = Matrix[[0],[0],[1],[1]]

syn0 = Matrix[[1 * Random.new(1).rand(2.0) - 1], 
              [1 * Random.new(1).rand(2.0) - 1], 
              [1 * Random.new(1).rand(2.0) - 1]]
l1 = Matrix[]

for i in 1..10000
  l0 = x
  l1 = sigmoid(l0 * syn0)

  l1_error = y - l1

  l1_delta = matrix_term_multiply(l1_error, sigmoid(l1, true))

  syn0 += l0.transpose * l1_delta
end

puts l1