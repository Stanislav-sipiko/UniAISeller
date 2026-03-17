import asyncio
import sys
from kernel import UkrSellKernel

async def test():
    kernel = UkrSellKernel()
    await kernel.initialize()
    
    slug = 'luckydog'
    ctx = kernel.registry.get_context(slug)
    
    if not ctx:
        print(f"Магазин {slug} не найден")
        return

    query = "WAUDOG шлейка"
    # Прямой вызов поиска через движок ретривала
    res = await ctx.retrieval.search(query=query, top_k=5)
    
    print(f"\n🔍 Результаты ретривала для: {query}")
    for i, p in enumerate(res.get('products', []), 1):
        # В разных версиях структура может отличаться, пробуем достать данные
        data = p.get('data') or p.get('product') or p
        title = data.get('title') or data.get('name')
        print(f"{i}. {title} | Score: {p.get('score')}")
    
    await kernel.close()

asyncio.run(test())
