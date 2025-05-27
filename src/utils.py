import logging

def make_log_formatter():
    spc=' '
    formatter = logging.Formatter(
        f'%(asctime)s | %(levelname)s @ %(name)s\n{spc*20}| %(message)s', '%Y-%m-%d %H:%M:%S'
    )
    return formatter
    
def configure_loggers( loggers, logging_level, log_here= True, log_to= None):
    
    formatter = make_log_formatter()
    
    if log_here:
        chandler = logging.StreamHandler()
        chandler.setLevel( logging_level)
        chandler.setFormatter( formatter)
        _ = [x.addHandler( chandler) for x in loggers]
    
    if log_to is not None:
        phoebe = logging.FileHandler( log_to)
        phoebe.setLevel( logging_level)
        phoebe.setFormatter( formatter)
        _ = [x.addHandler( phoebe) for x in loggers]        
    elif not log_here:
        fname = inspect.stack()[0].function
        print(f'{fname} did not add any handlers!')
    
    _ = [x.setLevel( logging_level) for x in loggers]
    
    return loggers

def filter_di( dict_, func_):
    return {k:v for k,v in dict_.items() if func_((k,v))}

def filter_li( list_, func_):
    return [l for l in list_ if func_(l)]