def Model_config():
    config = {}
    config['batch_size'] = 32
    config['num_workers'] = 8                 # number of data loading workers
    config['epochs'] = 200                    # number of total epochs to run
    config['lr'] = 1e-4                       # initial learning rate
    config['num_classes'] = 86
    
    # Graphormer
    config['n_layers'] = 3
    config['num_heads'] = 8

    config['hidden_dim'] = 256
    config['fnn_dim'] = 256

    config['input_dropout_rate'] = 0
    config['encoder_dropout_rate'] = 0
    config['attention_dropout_rate'] = 0
    
    config['flatten_dim'] = 2048              # molormer:2048, transformer:8192， no_distil：8192， no_share:2048
    
    return config
