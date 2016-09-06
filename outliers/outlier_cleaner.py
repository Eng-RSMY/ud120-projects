#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    import operator
    residual = (net_worths - predictions) ** 2
    sorted_residual = sorted( enumerate(residual), key=operator.itemgetter(1))
    # sorted_residual = sorted(enumerate(residual), key=lambda x: x[1])
    new_idx = list(zip(*sorted_residual[:-len(predictions)//10]))[0]
    ages = ages[[new_idx]]
    net_worths = net_worths[[new_idx]]
    residual = residual[[new_idx]]
    cleaned_data = list(zip(ages, net_worths, residual))


    ### your code goes here

    
    return cleaned_data

