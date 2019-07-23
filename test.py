
import numpy as np 
import pandas as pd 
orders = pd.DataFrame(np.array([[5,583,'2017-01-01 15:03:17'], [13,900,'2019-02-05 05:02:59' ], [69,19573,'2018-11-03 23:59:59'],[15,900,'2019-02-13 13:00:13']]), 
                      columns = ['OrderId', 'CustomerId', 'DateTime'])
order_lines = pd.DataFrame(np.array([[5873,5,3026.0],[7265,5,573.0],[9675,5,159.0], [5873,6,2999.0], [13,6,57.0],[5873,13,3026.0],[7265,13,573.0],[5873,15,3026.0],[7265,15,573.0]]),
                           columns = ['ProductId','OrderId','Price'])
orders['DateTime'] = pd.to_datetime(orders.DateTime)
orders['Month'] = pd.DatetimeIndex(orders['DateTime']).month
orders['Year'] = pd.DatetimeIndex(orders['DateTime']).year
n = 2

def report(orders, order_lines, n = 2):
    orders = orders.sort_values(by = ['Year', 'Month'], ascending = False).reset_index(drop=True)
    month = orders['Month'][0]
    year = orders['Year'][0]
    list_of_orders = orders[(orders.Month == month) & (orders.Year == year)]['OrderId']
    list_of_products = order_lines[order_lines.OrderId.isin(list_of_orders)].groupby(['ProductId'])['Price'].sum().reset_index()
    popular_products = order_lines[order_lines.OrderId.isin(list_of_orders)].groupby(['ProductId']).size().reset_index().rename(columns = {0:'Count'}).nlargest(n,columns = 'Count')
    profit_products = order_lines[order_lines.ProductId.isin(popular_products.ProductId)].groupby(['ProductId'])['Price'].sum().reset_index() 
    avg_of_orders = order_lines[order_lines.ProductId.isin(list_of_products.ProductId)].groupby(['OrderId'])['Price'].mean().reset_index()
    avg_of_orders = avg_of_orders[avg_of_orders.OrderId.isin(orders.OrderId)]
    print('Popular products of last month \n', popular_products)
    print('Profit of popular items for all time \n', profit_products)
    print('Average by order \n', avg_of_orders)
    
report(orders, order_lines, n)