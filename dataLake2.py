from pandasai import SmartDataframe
import pandas as pd
from pandasai.llm import OpenAI
df = pd.DataFrame({
    "country": [
        "United States", "United Kingdom", "France", "Germany", "Italy", "Spain", "Canada", "Australia", "Japan", "China"],
    "gdp": [
        19294482071552, 2891615567872, 2411255037952, 3435817336832, 1745433788416, 1181205135360, 1607402389504, 1490967855104, 4380756541440, 14631844184064
    ],
})
llm = OpenAI('sk-YmucvyKpQJMnyjuXagmvTSdkDALwlQJ9sWBsFrKiJ1FuAMo3J')
df = SmartDataframe([df],config={"llm":llm})
df.chat('Which are the countries with GDP greater than 3000000000000?')