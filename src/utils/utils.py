# origin: https://stackoverflow.com/a/30629776
def convert_to_float(frac_str):
    try:
        return float(frac_str)
    except ValueError:
        print(frac_str)
        num, denom = frac_str.split('/')
        try:
            leading, num = num.split(' ')
            whole = float(leading)
        except ValueError:
            whole = 0
        frac = float(num) / float(denom)
        return whole - frac if whole < 0 else whole + frac


def check_mozart_spelling(composers_list):
    for i, composer in enumerate(composers_list):
        if composer == 'mozzart':
            print("Pizza has 2 z's, not Mozart")
            composers_list[i] = 'mozart'
            break
    return composers_list
