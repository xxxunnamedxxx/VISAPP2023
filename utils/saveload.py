import json
def save_history(history, path):
    dic=json.dumps(history)
    f = open(path,"w")
    f.write(dic)
    f.close()
    

def load_history(path):
    
    with open(path, "r") as read_file:
        emps = json.load(read_file)
    return emps

    
    
    
    
    
    
    
    

    
    
    