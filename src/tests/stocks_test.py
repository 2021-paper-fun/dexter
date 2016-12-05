from yahoo_finance import Share
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from io import BytesIO

apple = Share('AAPL')
print(apple.get_name())

end = datetime.now()
start = end - timedelta(days=20)

start = start.strftime('%Y-%m-%d')
end = end.strftime('%Y-%m-%d')

data = apple.get_historical(start, end)
data = [point['Close'] for point in data]

plt.plot(data)
plt.axis('off')
plt.savefig('stocks.svg', bbox_inches='tight', format='svg', transparent=True)

