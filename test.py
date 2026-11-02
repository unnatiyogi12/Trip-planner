from tools.tavily_tool import tavily_search
from tools.flight_tool import search_flights

res = tavily_search("Plan a 7 days India to USA")
print(res)