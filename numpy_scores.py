# =========================================
# TASK 1 — Generate and Inspect the Data
# =========================================

np.random.seed(42)

# Generate scores (5 students, 4 subjects)
scores = np.random.randint(50, 101, size=(5, 4))

print("Scores:\n", scores)

# 3rd student, 2nd subject (index [2,1])
print("\n3rd student, 2nd subject:", scores[2, 1])

# Last 2 students (all subjects)
print("\nLast 2 students:\n", scores[-2:, :])

# First 3 students, subjects 2 and 3
print("\nFirst 3 students, subjects 2 & 3:\n", scores[:3, 1:3])


# =========================================
# TASK 2 — Analyze with Broadcasting
# =========================================

# Column-wise mean (per subject)
col_mean = np.round(np.mean(scores, axis=0), 2)
print("\nColumn-wise Mean:", col_mean)

# Add curve using broadcasting
curve = np.array([5, 3, 7, 2])
curved_scores = scores + curve

# Ensure scores do not exceed 100
curved_scores = np.clip(curved_scores, None, 100)

print("\nCurved Scores:\n", curved_scores)

# Row-wise max (best subject per student)
row_max = np.max(curved_scores, axis=1)
print("\nBest score per student:", row_max)


# =========================================
# TASK 3 — Normalize and Identify
# =========================================

# Min-max normalization per row
row_min = np.min(curved_scores, axis=1, keepdims=True)
row_max = np.max(curved_scores, axis=1, keepdims=True)

normalized = (curved_scores - row_min) / (row_max - row_min)

print("\nNormalized Scores:\n", normalized)

# Find global maximum in normalized array
max_position = np.unravel_index(np.argmax(normalized), normalized.shape)

print("\nHighest normalized score at:")
print("Student index:", max_position[0])
print("Subject index:", max_position[1])

# Extract scores strictly above 90
above_90 = curved_scores[curved_scores > 90]
print("\nScores above 90:", above_90)