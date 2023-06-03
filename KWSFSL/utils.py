def filter_opt(opt, tag):
    ret = { }

    for k,v in opt.items():
        tokens = k.split('.')
        if tokens[0] == tag:
            ret['.'.join(tokens[1:])] = v

    return ret


