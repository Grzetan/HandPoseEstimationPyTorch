architecture = [
    ('Dense', 'IBL', 8),
    ('Transition', 64),
    ('Dense', 'IBL', 8),
    ('Transition', 64),
    # ('Dense', 'AAIBL', 6),
    # ('Transition', 64),
    # ('Dense', 'AAIBL', 8),
    # ('Transition', 64),
    # ('Dense', 'AAIBL', 10),
    # ('Transition', 64),
    # ('Dense', 'AAIBL', 12),
    # ('Transition', 128),
    # ('Dense', 'AAIBL', 14),
    # ('Transition', 128),
    # ('Dense', 'AAIBL', 32),
    # ('AAIBL'),
    ('AvgPool', 2, 2), # kernel_size, stride
    ('out', 42) # 42 = n_points * 2
]