
def getArity(name):
    if name in ["x","-1","0","1","0.5","0.1","pi","e"]:
        return 0
    elif name in ["sin","cos","tan","log","exp","sqrt"]:
        return 1
    elif name in ["add","sub","mul","div"]:
        return 2
    else:
        print("bug : not define ",name)
        exit()

def addArityInfo(array):
    new_array = [(name,getArity(name)) for name in array]
    return new_array

def transformTexForm(array):
    def getEnd(start,array):
        _sum = 1
        for i in range(start,len(array)):
            arity = array[i][1]
            _sum += arity -1
            if _sum == 0:
                return i+1
    mid_end_indexs = []
    loopflg = True
    while loopflg:
        loopflg = False
        for i in range(len(array)):
            name,arity = array[i]
            if arity == 2:
                loopflg = True
                end = getEnd(i,array)
                mid = getEnd(i+1,array)
                if name == "add":
                    array.insert(mid,("+",1))
                    del array[i]
                if name == "sub":
                    array.insert(mid,("-",1))
                    del array[i]
                if name == "mul":
                    array.insert(mid,(" ",1))
                    del array[i]
                if name == "div":
                    array[i] = ("frac",1)
                break
    for i in range(len(array)):
        if array[i][0] == "frac":
            array[i] = ("\\frac",2)
        if array[i][0] == "pi":
            array[i] = ("\\pi",0)
    str_arr = [""]
    for node in array:
        name,arity = node
        for i in range(len(str_arr)):
            if str_arr[i] == "":
                if arity == 0:
                    str_arr = str_arr[:i] + [name] + str_arr[i+1:]
                if arity == 1:
                    if name == "sqrt":
                        name = "\\"+name
                        str_arr = str_arr[:i] + [name,"{","","}"]+ str_arr[i+1:]
                    else:
                        str_arr = str_arr[:i] + [name,"(","",")"]+ str_arr[i+1:]
                if arity == 2:
                    temp = [name,"{","","}{","","}"]
                    str_arr = str_arr[:i] + temp+ str_arr[i+1:]
                break
            if "" not in str_arr:
                str_arr.append(name)
                str_arr.append("")
    return "".join(str_arr)

string = "div(sqrt(sub(x,1)),tan(0.5))"
array = []
s = ""
for c in string:
    if c in "(),":
        if s != "":
            array.append(s)
        s = ""
    else:
        s+=c
array = addArityInfo(array)
print(transformTexForm(array))