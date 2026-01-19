"""
test_tools.py
Test individual tools without running the full agent
"""

import json
from tools import sql_query, create_visualization, statistical_analysis, save_insight


def test_sql_query():
    """Test the SQL query tool"""
    print("\n" + "=" * 80)
    print("🧪 Testing SQL Query Tool")
    print("=" * 80 + "\n")

    # Test 1: Get all brands
    print("Test 1: Get all brands")
    result = sql_query.invoke({
        "query": "SELECT DISTINCT brand FROM emails LIMIT 10"
    })
    data = json.loads(result)
    print(json.dumps(data, indent=2))

    # Test 2: Average discount by brand
    print("\n" + "-" * 80)
    print("Test 2: Average discount by brand")
    result = sql_query.invoke({
        "query": """
            SELECT brand, 
                   COUNT(*) as email_count,
                   AVG(discount_percent) as avg_discount,
                   MAX(discount_percent) as max_discount
            FROM emails 
            WHERE discount_percent IS NOT NULL
            GROUP BY brand
            ORDER BY avg_discount DESC
            LIMIT 5
        """
    })
    data = json.loads(result)
    print(json.dumps(data, indent=2))

    # Test 3: Security test (should fail)
    print("\n" + "-" * 80)
    print("Test 3: Security test (should reject DELETE)")
    result = sql_query.invoke({
        "query": "DELETE FROM emails WHERE id = '123'"
    })
    data = json.loads(result)
    print(json.dumps(data, indent=2))


def test_visualization():
    """Test the visualization tool"""
    print("\n" + "=" * 80)
    print("🧪 Testing Visualization Tool")
    print("=" * 80 + "\n")

    result = create_visualization.invoke({
        "viz_type": "bar",
        "data_query": """
            SELECT brand, AVG(discount_percent) as avg_discount
            FROM emails
            WHERE discount_percent IS NOT NULL
            GROUP BY brand
            ORDER BY avg_discount DESC
            LIMIT 10
        """,
        "title": "Average Discount by Brand",
        "x_column": "brand",
        "y_column": "avg_discount"
    })

    data = json.loads(result)
    print(json.dumps(data, indent=2))

    if data.get("success"):
        print(f"\n✅ Chart created: {data['viz_path']}")
        print("   Open in browser to view!")


def test_statistical_analysis():
    """Test the statistical analysis tool"""
    print("\n" + "=" * 80)
    print("🧪 Testing Statistical Analysis Tool")
    print("=" * 80 + "\n")

    print("Test 1: Distribution of discounts by brand")
    result = statistical_analysis.invoke({
        "data_query": """
            SELECT brand, discount_percent
            FROM emails
            WHERE discount_percent IS NOT NULL
        """,
        "analysis_type": "distribution",
        "column": "discount_percent",
        "group_by": "brand"
    })
    data = json.loads(result)
    print(json.dumps(data, indent=2)[:500] + "...")


def test_save_insight():
    """Test the save insight tool"""
    print("\n" + "=" * 80)
    print("🧪 Testing Save Insight Tool")
    print("=" * 80 + "\n")

    result = save_insight.invoke({
        "category": "discount_trend",
        "finding": "Banana Republic shows increasing discount trend",
        "metric_value": 35.5,
        "metric_name": "discount_increase_pct",
        "confidence": "high",
        "supporting_data": json.dumps({
            "sample_size": 50,
            "p_value": 0.001
        })
    })

    data = json.loads(result)
    print(json.dumps(data, indent=2))

    # Verify it was saved
    print("\n" + "-" * 80)
    print("Verifying saved insight:")
    result = sql_query.invoke({
        "query": "SELECT * FROM generated_insights ORDER BY id DESC LIMIT 1"
    })
    data = json.loads(result)
    print(json.dumps(data, indent=2))


def run_all_tests():
    """Run all tool tests"""
    print("\n" + "=" * 80)
    print("🔬 RUNNING ALL TOOL TESTS")
    print("=" * 80)

    test_sql_query()
    test_visualization()
    test_statistical_analysis()
    test_save_insight()

    print("\n" + "=" * 80)
    print("✅ ALL TESTS COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_all_tests()
