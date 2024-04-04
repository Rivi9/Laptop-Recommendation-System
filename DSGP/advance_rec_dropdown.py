import pandas as pd
import difflib

laptops_df = pd.read_csv("laptops_cleaned.csv")

def advance_rec_dropdown(price, ram, gpu):
    recommendation = recommend_laptop_dropdown(price, ram, gpu)
    recommendation = pd.DataFrame(recommendation)
    # if recommendation is None:
    #     return None
    # else:
    print(recommendation)
    recommended_laptops = []

    length = 10
    if recommendation.shape[0] < 10:
        length = recommendation.shape[0]
    for i in range(length):
        company = recommendation.iloc[i, recommendation.columns.get_loc('Company')]
        product = recommendation.iloc[i, recommendation.columns.get_loc('Product')]
        ram = recommendation.iloc[i, recommendation.columns.get_loc('Ram')]
        gpu = recommendation.iloc[i, recommendation.columns.get_loc('Gpu')]
        price = recommendation.iloc[i, recommendation.columns.get_loc('Price_euros')]
        recommended_laptops.append({'Company': company, 'Product': product, 'Ram': ram, 'Gpu': gpu, 'Price': price})
    return recommended_laptops


def recommend_laptop_dropdown(price, ram, gpu):
    price = float(price)
    ram = int(ram)
    """Recommend a laptop based on price, RAM, and GPU."""
    # Filter laptops based on user input
    filtered_laptops = laptops_df
    if price:
        filtered_laptops = filtered_laptops[filtered_laptops['Price_euros'] <= price]
    if ram:
        filtered_laptops = filtered_laptops[filtered_laptops['Ram'] >= ram]
    if gpu:
        filtered_laptops = filtered_laptops[filtered_laptops['Gpu'] == gpu]

    #print(len(filtered_laptops))
    # Sort laptops by price and return the top recommendation
    if not filtered_laptops.empty:
        return filtered_laptops.sort_values(by='Price_euros')
        #return filtered_laptops
    #if there are no results for the given inputs then the system will use the similar GPUs
    else:
        similar_gpus = find_similar_gpu(gpu, gpus_list)
        recommendation = recommend_laptop_nlp(price, ram, gpu, similar_gpus)
        return recommendation
        #return "No laptops match the specified criteria."


gpus_list = laptops_df['Gpu'].unique().tolist()
def find_similar_gpu(user_gpu, gpus_list, threshold=0.8):
    """
    Find GPUs in the list that are similar to the user-specified GPU.
    """
    similar_gpus = []
    words = user_gpu.split()
    for gpu in gpus_list:
        similarity = difflib.SequenceMatcher(None, user_gpu.lower(), gpu.lower()).ratio()
        if len(words) == 1:
            if similarity >= 0.3:
                similar_gpus.append(gpu)
        elif len(words) == 2:
            if similarity >= 0.5:
                similar_gpus.append(gpu)
        elif len(words) == 3:
            if similarity >= 0.8:
                similar_gpus.append(gpu)
        else:
            if similarity >= 0.9:
                similar_gpus.append(gpu)
    return similar_gpus

def recommend_laptop_nlp(price, ram, gpu, similar_gpus):
    """Recommend a laptop based on price, RAM, and GPU."""
    # Filter laptops based on user input
    filtered_laptops = laptops_df
    if price:
        filtered_laptops = filtered_laptops[filtered_laptops['Price_euros'] <= price]
    if ram:
        filtered_laptops = filtered_laptops[filtered_laptops['Ram'] >= ram]
    if gpu:
        if not len(similar_gpus) == 0:
            filtered_laptops = filtered_laptops[filtered_laptops['Gpu'].isin(similar_gpus)]
        else:
            similar_gpus1 = []
            words = gpu.split()
            user_gpu = words[0]
            for gpu in gpus_list:
                similarity = difflib.SequenceMatcher(None, user_gpu.lower(), gpu.lower()).ratio()
                if similarity >= 0.4:
                    similar_gpus1.append(gpu)
            filtered_laptops = filtered_laptops[filtered_laptops['Gpu'].isin(similar_gpus1)]
    #print(len(filtered_laptops))
    # Sort laptops by price and return the top recommendation
    if not filtered_laptops.empty:
        print(len(filtered_laptops))
        return filtered_laptops.sort_values(by='Price_euros')
        #return filtered_laptops.sort_values(by='Price_euros')
        #return filtered_laptops
    else:
        return None