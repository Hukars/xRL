from offlinerl import algos

def select_algo(algo_name):
    algo_name = algo_name.upper()
    assert algo_name in algos.__all__ 
    
    return eval("algos."+algo_name)