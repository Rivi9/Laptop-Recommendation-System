import pandas as pd
import difflib

laptops_df = pd.read_csv("laptops_cleaned.csv")

# price = 2339.69
# ram = 8
# gpu = "Intel iris plus graphics 640"
# cpu = "Intel Core i7 g550U 1.85Hz"

def advance_rec_dropdown(price, ram, gpu, cpu):
    recommendation = recommend_laptop_dropdown(price, ram, gpu, cpu)
    recommendation = pd.DataFrame(recommendation)
    print(recommendation)
    recommended_laptops = []

    length = 10
    if recommendation.shape[0] < 10:
        length = recommendation.shape[0]
    for i in range(length):
        company = recommendation.iloc[i, recommendation.columns.get_loc('Company')]
        product = recommendation.iloc[i, recommendation.columns.get_loc('Product')]
        ram = recommendation.iloc[i, recommendation.columns.get_loc('Ram')]
        cpu = recommendation.iloc[i, recommendation.columns.get_loc('Cpu')]
        gpu = recommendation.iloc[i, recommendation.columns.get_loc('Gpu')]
        price = recommendation.iloc[i, recommendation.columns.get_loc('Price')]
        recommended_laptops.append({'Company': company, 'Product': product, 'Ram': ram, 'Cpu': cpu, 'Gpu': gpu, 'Price': price})
    return recommended_laptops

def recommend_laptop_dropdown(price, ram, gpu, cpu):
    """
    1st attempt
    """
    price = float(price)
    ram = int(ram)
    """Recommend a laptop based on price, RAM, and GPU."""
    # Filter laptops based on user input
    filtered_laptops = laptops_df
    if price:
        filtered_laptops = filtered_laptops[filtered_laptops['Price'] <= price]
    if ram:
        filtered_laptops = filtered_laptops[filtered_laptops['Ram'] >= ram]
    if gpu:
        filtered_laptops = filtered_laptops[filtered_laptops['Gpu'] == gpu]
    if cpu:
        filtered_laptops = filtered_laptops[filtered_laptops['Cpu'] == cpu]
    # Sort laptops by price and return the top recommendation
    if not filtered_laptops.empty:
        return filtered_laptops.sort_values(by='Price')
        #return filtered_laptops
    #if there are no results for the given inputs then the system will use the similar GPUs and similar CPUs
    else:
        #if there are no laptops for the given exact inputs  it looks for something with a similar GPU and CPU
        similar_gpus = find_similar_gpu(gpu, gpus_list)
        similar_cpus = find_similar_cpu(cpu, cpus_list, 0.8)
        recommendation = recommend_laptop_try1(price, ram, gpu, cpu, similar_gpus, similar_cpus)
        return recommendation
        #return "No laptops match the specified criteria."


gpus_list = laptops_df['Gpu'].unique().tolist()
cpus_list = laptops_df['Cpu'].unique().tolist()
def find_similar_gpu(user_gpu, gpus_list):
    """
    Find GPUs in the list that are similar to the user-specified GPU.
    """
    similar_gpus = []
    words = user_gpu.split()
    for gpu in gpus_list:
        similarity = difflib.SequenceMatcher(None, user_gpu.lower(), gpu.lower()).ratio()
        if similarity >= 0.8:
            similar_gpus.append(gpu)
    return similar_gpus

def find_similar_cpu(user_cpu, cpus_list, similarity_value):
    """
    Find CPUs in the list that are similar to the user-specified GPU.
    """
    similar_cpus = []
    words = user_cpu.split()
    for cpu in cpus_list:
        similarity = difflib.SequenceMatcher(None, user_cpu.lower(), cpu.lower()).ratio()
        if similarity >= similarity_value:
            similar_cpus.append(cpu)
    print(similar_cpus)
    return similar_cpus

def recommend_laptop_without_gpu(price, ram, cpu, similar_cpus):
    print("without gpu")
    """Recommend a laptop based on price, RAM, and GPU."""
    # Filter laptops based on user input
    filtered_laptops = laptops_df
    if price:
        filtered_laptops = filtered_laptops[filtered_laptops['Price'] <= price]
    if ram:
        filtered_laptops = filtered_laptops[filtered_laptops['Ram'] >= ram]
    if cpu:
        if not len(similar_cpus) == 0:
            filtered_laptops = filtered_laptops[filtered_laptops['Cpu'].isin(similar_cpus)]
        # else:
        #     similar_cpus1 = []
        #     words = cpu.split()
        #     user_cpu = words[0]
        #     for cpu in cpus_list:
        #         similarity = difflib.SequenceMatcher(None, user_cpu.lower(), cpu.lower()).ratio()
        #         if similarity >= 0.2:
        #             similar_cpus1.append(cpu)
        #     print(similar_cpus1)
        #     filtered_laptops = filtered_laptops[filtered_laptops['Cpu'].isin(similar_cpus1)]


    if not filtered_laptops.empty:
        return filtered_laptops.sort_values(by='Price')
        #return filtered_laptops.sort_values(by='Price')
        #return filtered_laptops
    else:
        return None

def recommend_laptop_try1(price, ram, gpu, cpu, similar_gpus, similar_cpus):
    """Recommend a laptop based on price, RAM, and GPU."""
    # Filter laptops based on user input with similar GPUs and CPUs
    filtered_laptops = laptops_df
    if price:
        filtered_laptops = filtered_laptops[filtered_laptops['Price'] <= price]
    if ram:
        filtered_laptops = filtered_laptops[filtered_laptops['Ram'] >= ram]
    if gpu:
        filtered_laptops = filtered_laptops[filtered_laptops['Gpu'].isin(similar_gpus)]
    if cpu:
        filtered_laptops = filtered_laptops[filtered_laptops['Cpu'].isin(similar_cpus)]


    if not filtered_laptops.empty:
        print("with gpu")
        return filtered_laptops.sort_values(by='Price')
    else:
        # if there are no results for the given inputs then the system will use only the similar CPUs
        similar_cpus = find_similar_cpu(cpu, cpus_list, 0.9)
        recommendation = recommend_laptop_without_gpu(price, ram, cpu, similar_cpus)
        return recommendation
# advance_rec_dropdown(price,ram,gpu)