# Frequently Asked Questions

### How do I get started?
It really depends on what you're aiming for. 

If you're looking to connect your AI application to the internet with our tailored API, check out the [Python](/docs/python-sdk/getting-started) or [REST API](/docs/rest-api/api-reference) documentation. 
You can also check out demos and examples for inspiration [here](/docs/python-sdk/examples).

If you're looking to build and deploy our open source autonomous research agent GPT Researcher, please see [GPT Researcher](/docs/gpt-researcher/introduction) documentation.

### How do you ensure the report is factual and accurate?
We do our best to ensure that the information we provide is factual and accurate. We do this by using multiple sources, and by using proprietary AI to score and rank the most relevant and accurate information. We also use proprietary AI to filter out irrelevant information and sources.

Lastly, by using RAG and other techniques, we ensure that the information is relevant to the context of the research task, leading to more accurate generative AI content and reduced hallucinations.

### What is Tavily API?
Tavily search API is a search engine optimized for LLMs, aimed at efficient, quick and persistent search results. Unlike other search APIs such as Serp or Google, Tavily focuses on optimizing search for AI developers and autonomous AI agents. We take care of all the burden in searching, scraping, filtering and extracting the most relevant information from online sources. All in a single API call!

The search API can also be used return answers to questions (for use cases such as multi-agent frameworks like autogen) and can complete comprehensive research tasks in seconds. Moreover, Tavily leverages proprietary news, weather, and other internal data sources to complement online information.

To try the API in action, you can now use our hosted version [here](https://app.tavily.com/chat) or on our [API Playground](https://app.tavily.com/playground).
### How is Tavily different from other search APIs?
Current search APIs such as Google, Serp and Bing retrieve search results based on user query. However, the results are sometimes irrelevant to the goal of the search, and return simple site URLs and snippets of content which are not always relevant. Because of this, any developer would need to then scrape the sites for relevant content, filter irrelevant information, optimize the content to fit LLM context limits, and more. This tasks is a burden and requires skills to get right.

Tavily Search API aggregates up to 20 sites per a single API call, and uses AI to score, filter and rank the top most relevant sources and content to your task, query or goal. In addition, Tavily allows developers to add custom fields such as context and limit response tokens to enable the optimal search experience for LLMs.
Lastly, Tavily indexes and ranks search results based on factors such as trusted sources, content quality, and more. This allows for a more accurate and relevant search experience for AI agents.

Remember: With LLM hallucinations, it's crucial to optimize for RAG with the right context and information.

### What is the Tavily API pricing?
Tavily is free to use for up to 1,000 API calls per month. Check out our [pricing page](https://tavily.com/#pricing) to see our other pricing plans.

### What are your plans for the future?
We're constantly working on improving our products and services. We're currently working on improving our search API together with design partners, and adding more data sources to our search engine.

Feel free to [contact us](mailto:support@tavily.com) if you have any further questions or suggestions!

### What is GPT Researcher?
GPT Researcher is a popular open source autonomous research agent that takes care of the tedious task of research for you, by scraping, filtering and aggregating up to 20 web sources per a single research task.

GPT Researcher is built with best practices for leveraging LLMs (prompt engineering, RAG, chains, embeddings, etc), and is optimized for quick and efficient research. It is also fully customizable and can be tailored to your specific needs.

To learn more about GPT Researcher, check out the [documentation page](/docs/gpt-researcher/introduction).
### How much does each research run cost?
A research task using GPT Researcher costs around $0.01 per a single run (for GPT-4 usage). We're constantly optimizing LLM calls to reduce costs and improve performance. 