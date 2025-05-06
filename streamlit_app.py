import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import os
# Import the main google.genai module for the new SDK
from google import genai

# --- Configuration Constants ---
# Define salary ranges and their recommended savings targets
SALARY_RANGES = {
    "0-10000": {"min": 0, "max": 10000, "savings_target": 0.10},
    "10000-20000": {"min": 10000, "max": 20000, "savings_target": 0.15},
    "20000-30000": {"min": 20000, "max": 30000, "savings_target": 0.20},
    "30000-40000": {"min": 30000, "max": 40000, "savings_target": 0.25},
    "40000-50000": {"min": 40000, "max": 50000, "savings_target": 0.30},
    "50000+": {"min": 50000, "max": float('inf'), "savings_target": 0.35}
}

# Initial Default Budget Categories and amounts (these will be stored in session state)
DEFAULT_BUDGET_CATEGORIES = {
    "Groceries": 5000,
    "Utilities": 3000,
    "Rent/Mortgage": 15000,
    "Transportation": 2000,
    "Dining Out": 3000,
    "Entertainment": 2000,
    "Shopping": 4000,
    "Health": 5000,
    "Other": 3000
}

# Simplified Tax Slabs (New Tax Regime for Individuals below 60 - FY 2024-25 / AY 2025-26)
# This is a simplified model and does not include standard deduction, cess, or other complexities.
TAX_SLABS = [
    (0, 300000, 0.0),      # 0 to 3 Lakhs: 0%
    (300001, 600000, 0.05), # 3 Lakhs to 6 Lakhs: 5% on income above 3 Lakhs
    (600001, 900000, 0.10), # 6 Lakhs to 9 Lakhs: 10% on income above 6 Lakhs
    (900001, 1200000, 0.15),# 9 Lakhs to 12 Lakhs: 15% on income above 9 Lakhs
    (1200001, 1500000, 0.20),# 12 Lakhs to 15 Lakhs: 20% on income above 12 Lakhs
    (1500001, float('inf'), 0.30) # Above 15 Lakhs: 30% on income above 15 Lakhs
]


# --- Helper Functions ---

def calculate_simplified_tax(annual_income):
    """
    Calculates a simplified estimated annual income tax based on predefined slabs.
    This is a basic calculation and does not account for all tax rules, deductions, or cess.
    """
    tax_amount = 0
    for lower_bound, upper_bound, rate in TAX_SLABS:
        if annual_income > lower_bound:
            # Calculate taxable amount within this slab
            taxable_in_slab = min(annual_income, upper_bound) - lower_bound
            tax_amount += taxable_in_slab * rate
        if annual_income <= upper_bound:
            break # Stop if we have passed the relevant slabs

    # Note: This simplified function does not include the tax rebate for income up to 7 lakhs in the new regime.
    # For a more accurate calculation, that rebate would need to be added.
    return tax_amount

def calculate_budget_metrics(df, budget_categories):
    """
    Calculate budget-related metrics for the current month.
    Compares actual spending against predefined budget amounts per category.
    Takes budget_categories as an argument (now from session state).
    """
    # Handle case with no data gracefully
    if df.empty:
        metrics = {
            'total_budget': sum(budget_categories.values()),
            'total_spent': 0,
            'category_metrics': {}
        }
        for category, budget in budget_categories.items():
             metrics['category_metrics'][category] = {
                'budget': budget,
                'spent': 0,
                'remaining': budget,
                'percentage': (0 if budget == 0 else -spent/budget*100) # Handle 0 budget, show negative percentage if overspent
             }
        return metrics

    current_month = datetime.now().month
    current_year = datetime.now().year

    # Filter DataFrame for the current month and year
    monthly_df = df[
        (df['Date'].dt.month == current_month) &
        (df['Date'].dt.year == current_year)
    ]

    metrics = {
        'total_budget': sum(budget_categories.values()), # Sum of all category budgets
        'total_spent': monthly_df['Amount'].sum(), # Total spending this month
        'category_metrics': {} # Dictionary to hold metrics per category
    }

    # Calculate spent, remaining, and percentage for each category
    for category, budget in budget_categories.items():
        spent = monthly_df[monthly_df['Category'] == category]['Amount'].sum()
        metrics['category_metrics'][category] = {
            'budget': budget,
            'spent': spent,
            'remaining': budget - spent,
            # Calculate percentage relative to budget, handle overspending
            'percentage': (spent / budget * 100) if budget > 0 else (100 if spent > 0 else 0) # Handle 0 budget, show 100% if spent
        }

    return metrics

def create_budget_vs_actual_chart(budget_metrics):
    """
    Create a grouped bar chart comparing the budgeted amount vs. the actual
    amount spent for each category in the current month.
    """
    data = []
    # Prepare data in a format suitable for Plotly Express
    for category, values in budget_metrics['category_metrics'].items():
        data.append({'Category': category, 'Type': 'Budget', 'Amount': values['budget']})
        data.append({'Category': category, 'Type': 'Spent', 'Amount': values['spent']})

    df_chart = pd.DataFrame(data)

    # Create the grouped bar chart
    fig = px.bar(df_chart, x='Category', y='Amount', color='Type', barmode='group',
                 title='Budget vs. Actual Spending by Category (Current Month)',
                 color_discrete_map={'Budget': 'skyblue', 'Spent': 'salmon'}) # Assign specific colors

    fig.update_layout(
        yaxis_title='Amount (₹)',
        xaxis_title='Category',
        hovermode='x unified' # Improve hover experience
    )

    return fig

def create_savings_analysis_chart(monthly_salary, total_spent, monthly_tax, salary_range_info):
    """
    Create a bar chart comparing the recommended savings based on salary range
    vs. the actual savings (Income - Total Spent - Estimated Tax).
    """
    recommended_savings = monthly_salary * salary_range_info["savings_target"]
    # Calculate actual savings considering estimated tax
    actual_savings = max(0, monthly_salary - total_spent - monthly_tax) # Savings cannot be negative

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=['Recommended Savings', 'Actual Savings (Post-Tax)'], # Updated label
        y=[recommended_savings, actual_savings],
        text=[f'₹{recommended_savings:,.0f}', f'₹{actual_savings:,.0f}'], # Display values on bars
        textposition='auto',
        marker_color=['skyblue', 'lightgreen' if actual_savings >= recommended_savings else 'salmon'] # Color based on performance
    ))

    fig.update_layout(
        title='Monthly Savings Performance (Post-Tax)', # Updated title
        yaxis_title='Amount (₹)',
        showlegend=False, # No legend needed for two bars
        hovermode='x unified'
    )

    return fig

def create_category_pie_chart(df):
    """
    Create a pie chart showing the distribution of spending across different
    categories using all recorded data.
    """
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title='Spending by Category (No Data)')
        return fig

    # Group by category and sum the amounts
    category_spending = df.groupby('Category')['Amount'].sum().reset_index()

    # Create the pie chart
    fig = px.pie(
        category_spending,
        values='Amount',
        names='Category',
        title="Spending Distribution by Category (All Time)",
        hole=0.4 # Create a donut chart
    )
    fig.update_traces(
        textposition='inside', # Place text inside slices
        textinfo='percent+label', # Show percentage and label
        insidetextorientation='radial' # Orient text radially
    )
    return fig

def create_monthly_spending_trend_chart(df):
    """
    Create a line chart showing the trend of total spending over months.
    """
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title='Monthly Spending Trend (No Data)')
        return fig

    # Ensure 'Date' is datetime and extract Month-Year for grouping
    df['MonthYear'] = df['Date'].dt.to_period('M').astype(str)
    monthly_spending = df.groupby('MonthYear')['Amount'].sum().reset_index()

    # Create a helper column for sorting chronologically
    monthly_spending['SortKey'] = pd.to_datetime(monthly_spending['MonthYear'])
    monthly_spending = monthly_spending.sort_values('SortKey') # Sort by date

    # Create the line chart
    fig = px.line(
        monthly_spending,
        x='MonthYear',
        y='Amount',
        title="Monthly Spending Trend",
        labels={'Amount': 'Amount (₹)', 'MonthYear': 'Month'},
        markers=True # Add markers for data points
    )
    fig.update_layout(
        xaxis_tickangle=-45, # Angle x-axis labels for readability
        hovermode='x unified'
    )
    return fig

def create_income_vs_expense_trend_chart(df, monthly_salary):
    """
    Create a line chart comparing monthly income vs. monthly expenses over time.
    Assumes income is constant each month for simplicity, but could be extended.
    """
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title='Income vs. Expense Trend (No Data)')
        return fig

    # Ensure 'Date' is datetime and extract Month-Year
    df['MonthYear'] = df['Date'].dt.to_period('M').astype(str)
    monthly_spending = df.groupby('MonthYear')['Amount'].sum().reset_index()

    # Create a helper column for sorting chronologically
    monthly_spending['SortKey'] = pd.to_datetime(monthly_spending['MonthYear'])
    monthly_spending = monthly_spending.sort_values('SortKey') # Sort by date

    # Add monthly income to the DataFrame (assuming constant income for now)
    monthly_spending['Income'] = monthly_salary

    # Melt the DataFrame for Plotly Express to plot multiple lines
    # Changed value_name from 'Amount' to 'Value' to avoid potential conflict
    df_melted = monthly_spending.melt(
        id_vars=['MonthYear', 'SortKey'],
        value_vars=['Amount', 'Income'],
        var_name='Type',
        value_name='Value' # Changed from 'Amount'
    )

    # Create the line chart
    fig = px.line(
        df_melted.sort_values('SortKey'), # Sort the melted data for correct line drawing
        x='MonthYear',
        y='Value', # Use the new value column name
        color='Type',
        title="Monthly Income vs. Expenses Trend",
        labels={'Value': 'Amount (₹)', 'MonthYear': 'Month', 'Type': 'Category'}, # Update labels
        markers=True # Add markers for data points
    )
    fig.update_layout(
        xaxis_tickangle=-45, # Angle x-axis labels for readability
        hovermode='x unified'
    )
    return fig


def get_ai_financial_insights(monthly_salary, monthly_tax, budget_metrics, all_expenses_df, user_budget_categories):
    """
    Generate personalized financial insights and recommendations using the Gemini API.
    Analyzes current month's budget performance and overall spending trends, considering tax and user-defined budget.
    """
    try:
        # Configure Gemini API key from Streamlit secrets
        # This is the secure way to access your API key
        if "GOOGLE_API_KEY" not in st.secrets:
            return "AI insights are not configured. Please add your Google API key to `.streamlit/secrets.toml`."

        # Initialize the client with the API key for the new SDK
        client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])

        # --- Prepare Data for the AI Prompt ---
        total_spent_month = budget_metrics['total_spent']
        post_tax_income = monthly_salary - monthly_tax # Calculate post-tax income

        # Detail current month's budget vs. actual spending (using user-defined budget)
        category_budget_vs_actual = "Current Month Budget Performance (based on your set budget):\n"
        for category, data in budget_metrics['category_metrics'].items():
             category_budget_vs_actual += f"- {category}: Your Budget ₹{user_budget_categories.get(category, 0):,.2f}, Spent ₹{data['spent']:.2f}, Remaining ₹{data['remaining']:.2f}\n" # Use user budget

        # Summarize overall spending patterns from all time data
        all_time_category_spending = all_expenses_df.groupby('Category')['Amount'].sum().sort_values(ascending=False)
        all_time_spending_summary = "\nOverall spending across all recorded entries:\n"
        if not all_time_category_spending.empty:
             for category, amount in all_time_category_spending.items():
                  all_time_spending_summary += f"- {category}: ₹{amount:,.2f}\n"
        else:
             all_time_spending_summary += "No historical spending data recorded."

        # --- Craft the AI Prompt ---
        prompt = f"""Analyze the following personal finance data for a user in India.
        Provide personalized financial insights, actionable tips, and highlight areas for improvement in a concise manner using bullet points.
        Focus on the current month's performance compared to the **user-defined budget** and overall spending trends. Use Indian Rupees (₹) for currency.

        User's Monthly Gross Income: ₹{monthly_salary:,.2f}
        Estimated Monthly Tax: ₹{monthly_tax:,.2f}
        Estimated Monthly Post-Tax Income: ₹{post_tax_income:,.2f}
        Current Month Total Spent: ₹{total_spent_month:,.2f}
        {category_budget_vs_actual}
        {all_time_spending_summary}

        Based on this data, provide suggestions focusing on:
        1. Analysis of spending relative to **post-tax income** and **user-defined budget**.
        2. Identification of high spending categories relative to user-defined budget or overall spending.
        3. Tips to reduce spending in identified areas, considering the user's budget.
        4. General recommendations for improving savings and financial health, considering the tax burden and user's budget goals.
        5. Positive feedback where performance is good (e.g., staying within the user-defined budget in certain categories).
        6. Suggestions tailored to the income range and estimated tax impact.

        Keep the response encouraging, easy to understand, and formatted with bullet points. Start directly with the insights.
        """

        # --- Generate Content using the AI Model ---
        response = client.models.generate_content(model='gemini-2.0-flash', contents=prompt)

        # Return the text response from the AI
        return response.text

    except Exception as e:
        # Catch potential errors during API call
        st.error(f"An error occurred while fetching AI insights: {e}")
        return "Could not generate AI insights at this time."


# --- Streamlit App Layout and Logic ---

# Page Configuration: Set title, icon, and layout
st.set_page_config(page_title="Personal Finance Tracker", page_icon="💰", layout="wide")

# Initialize Session State: This dictionary persists across user interactions
if 'expenses_df' not in st.session_state:
    # Create an empty DataFrame if it doesn't exist
    st.session_state.expenses_df = pd.DataFrame(columns=['Date', 'Category', 'Amount', 'Description'])
    # Ensure 'Date' column is datetime type from the start
    st.session_state.expenses_df['Date'] = pd.to_datetime(st.session_state.expenses_df['Date'])

if 'currency_symbol' not in st.session_state:
    st.session_state.currency_symbol = '₹' # Default currency symbol

if 'salary_range' not in st.session_state:
    st.session_state.salary_range = list(SALARY_RANGES.keys())[0] # Default to the first salary range
if 'monthly_salary' not in st.session_state:
    # Default salary based on the initial salary range selection
    st.session_state.monthly_salary = float(SALARY_RANGES[st.session_state.salary_range]["min"])

# Initialize or load user-defined budget categories into session state
if 'user_budget_categories' not in st.session_state:
    st.session_state.user_budget_categories = DEFAULT_BUDGET_CATEGORIES.copy() # Copy default budgets initially


# --- Main App Title and Description ---
st.title("💰 Personal Finance Tracker")
st.markdown("Track your expenses, analyze spending patterns, and get personalized financial insights.")

# --- Sidebar for Settings (Income and Budget) ---
with st.sidebar:
    st.header("⚙️ Settings")

    st.subheader("💼 Income")
    # Select monthly salary range
    selected_salary_range = st.selectbox(
        "Select Monthly Salary Range",
        options=list(SALARY_RANGES.keys()),
        format_func=lambda x: f"₹{x.replace('-', ' - ')}", # Format display nicely
        key='salary_range_select' # Unique key for the widget
    )
    # Update session state only if the selected range changes
    if selected_salary_range != st.session_state.salary_range:
         st.session_state.salary_range = selected_salary_range
         # Optionally reset exact salary to the min of the new range
         st.session_state.monthly_salary = float(SALARY_RANGES[st.session_state.salary_range]["min"])


    range_info = SALARY_RANGES[st.session_state.salary_range]
    # Input for exact monthly income within the selected range
    exact_salary = st.number_input(
        "Enter Exact Monthly Income",
        min_value=float(range_info["min"]),
        # Set max value based on the range, use a large number for 50000+
        max_value=float(range_info["max"]) if range_info["max"] != float('inf') else 10000000.0,
        value=float(st.session_state.monthly_salary), # Use current session state value
        step=100.0, # Step value for input
        format="%.2f", # Format as currency
        key='exact_salary_input' # Unique key for the widget
    )
    # Update session state only if the exact salary input changes
    if exact_salary != st.session_state.monthly_salary:
        st.session_state.monthly_salary = exact_salary

    # --- Tax Settings ---
    st.subheader("📊 Tax Settings")
    st.write("Estimated tax is calculated using a simplified New Tax Regime.")
    # Could add options for Old Regime, deductions etc. here in the future

    # --- Customizable Budget Settings (Streamlined) ---
    st.subheader("📝 Set Your Monthly Budget")
    st.write("Select a category and set its monthly budget.")

    # Use a form for budget input to update a single category at a time
    with st.form("single_budget_form"):
        # Dropdown to select the category to budget for
        selected_budget_category = st.selectbox(
            "Select Category",
            options=list(st.session_state.user_budget_categories.keys()),
            key="selected_budget_category"
        )

        # Get the current budget for the selected category to pre-fill the input
        current_budget_amount = st.session_state.user_budget_categories.get(selected_budget_category, 0.0)

        # Number input for the budget amount
        new_budget_amount = st.number_input(
            f"Set Budget for {selected_budget_category}",
            min_value=0.0,
            step=100.0,
            format="%.2f",
            value=float(current_budget_amount), # Pre-fill with current budget
            key="new_budget_amount"
        )

        # Button to update the budget for the selected category
        update_budget_button = st.form_submit_button("Update Budget")

        # Logic to update the budget when the button is clicked
        if update_budget_button:
            st.session_state.user_budget_categories[selected_budget_category] = new_budget_amount
            st.success(f"✅ Budget for {selected_budget_category} updated to ₹{new_budget_amount:,.2f}!")

    # Display current budgets for reference
    st.markdown("---") # Separator
    st.write("**Current Monthly Budgets:**")
    for category, budget in st.session_state.user_budget_categories.items():
         st.write(f"- **{category}**: ₹{budget:,.2f}")


# --- Expense Input Form ---
st.subheader("📝 Add New Expense")
# Use a form to group inputs and clear them on submit
with st.form("expense_form", clear_on_submit=True):
    # Arrange input fields in columns for better layout
    cols = st.columns([1.5, 1.5, 1, 2]) # Adjust column widths

    with cols[0]:
        date = st.date_input("Date", value=datetime.now(), key="exp_date")
    with cols[1]:
        # Use the keys from the user's budget categories for the selectbox options
        category = st.selectbox("Category", options=list(st.session_state.user_budget_categories.keys()), key="exp_category")
    with cols[2]:
        amount = st.number_input("Amount", min_value=0.01, step=0.01, format="%.2f", key="exp_amount")
    with cols[3]:
        description = st.text_input("Description (Optional)", key="exp_description")

    # Submit button for the form
    submitted = st.form_submit_button("Add Expense")

    # Process form submission
    if submitted:
        # Create a new DataFrame row for the expense
        new_expense = pd.DataFrame([{
            'Date': pd.to_datetime(date), # Ensure datetime type for consistency
            'Category': category,
            'Amount': amount,
            'Description': description or "-" # Use "-" if description is empty
        }])
        # Concatenate the new expense to the existing DataFrame in session state
        st.session_state.expenses_df = pd.concat([st.session_state.expenses_df, new_expense], ignore_index=True)
        st.success("✅ Expense added successfully!")


# --- Data Processing and Display ---
st.subheader("📊 Financial Overview & Analysis")

# Only show analysis if there is data
if st.session_state.expenses_df.empty:
    st.info("👆 Start by adding your expenses using the form above to see your financial data.")
else:
    # Ensure 'Date' column is always datetime for consistent processing
    st.session_state.expenses_df['Date'] = pd.to_datetime(st.session_state.expenses_df['Date'])

    # Calculate budget metrics for the current month using user-defined budgets
    budget_metrics = calculate_budget_metrics(st.session_state.expenses_df, st.session_state.user_budget_categories)

    # Calculate estimated tax
    annual_income = st.session_state.monthly_salary * 12
    estimated_annual_tax = calculate_simplified_tax(annual_income)
    estimated_monthly_tax = estimated_annual_tax / 12 # Convert annual tax to monthly estimate
    post_tax_income = st.session_state.monthly_salary - estimated_monthly_tax # Calculate post-tax income


    # --- Overview Metrics (Current Month) ---
    st.subheader("Summary Metrics (Current Month)")
    col1, col2, col3, col4, col5 = st.columns(5) # Added one more column for tax

    with col1:
        st.metric("Total Income (Gross)", f"₹{st.session_state.monthly_salary:,.2f}")
    with col2:
         st.metric("Estimated Tax (Monthly)", f"₹{estimated_monthly_tax:,.2f}") # Display estimated tax
    with col3:
        st.metric("Total Expenses", f"₹{budget_metrics['total_spent']:,.2f}")
    with col4:
        # Calculate savings based on post-tax income
        savings = max(0.0, post_tax_income - budget_metrics['total_spent'])
        st.metric("Total Savings (Post-Tax)", f"₹{savings:,.2f}") # Updated label
    with col5:
        # Calculate savings rate based on post-tax income
        savings_rate = (savings / post_tax_income * 100) if post_tax_income > 0 else 0
        st.metric("Savings Rate (Post-Tax)", f"{savings_rate:.1f}%") # Updated label


    # --- Budget vs. Actual Chart ---
    st.subheader("📈 Budget Performance (Current Month)")
    # Pass budget_metrics which now uses user-defined budgets
    budget_chart = create_budget_vs_actual_chart(budget_metrics)
    st.plotly_chart(budget_chart, use_container_width=True) # Use container width for responsiveness

    # --- Savings Analysis Chart ---
    st.subheader("💰 Savings Analysis")
    # Get info for the selected salary range
    salary_info = SALARY_RANGES[st.session_state.salary_range]
    savings_chart = create_savings_analysis_chart(
        st.session_state.monthly_salary,
        budget_metrics['total_spent'],
        estimated_monthly_tax, # Pass estimated monthly tax
        salary_info
    )
    st.plotly_chart(savings_chart, use_container_width=True)

    # --- Expense Analysis Charts ---
    st.subheader("💳 Expense Breakdown & Trends")
    col_charts = st.columns(2) # Use columns to place charts side-by-side

    with col_charts[0]:
        # Category-wise spending pie chart (using all recorded data)
        category_pie = create_category_pie_chart(st.session_state.expenses_df)
        st.plotly_chart(category_pie, use_container_width=True)

    with col_charts[1]:
        # Monthly spending trend line chart (using all recorded data)
        monthly_trend = create_monthly_spending_trend_chart(st.session_state.expenses_df)
        st.plotly_chart(monthly_trend, use_container_width=True)

    # --- Income vs. Expense Trend Chart ---
    st.subheader("📊 Income vs. Expense Trend")
    # Note: This chart still uses Gross Income vs Expenses for simplicity in visualization.
    # A more complex chart could show Gross Income, Tax, and Expenses.
    income_expense_trend = create_income_vs_expense_trend_chart(st.session_state.expenses_df, st.session_state.monthly_salary)
    st.plotly_chart(income_expense_trend, use_container_width=True)


    # --- AI-Powered Financial Insights ---
    st.subheader("🤖 AI-Powered Financial Insights")
    st.write("Get personalized suggestions based on your spending data.")

    # Button to trigger AI insight generation (prevents API call on every rerun)
    if st.button("Generate AI Insights"):
        # Show a spinner while waiting for the AI response
        with st.spinner("Generating insights..."):
            ai_insights = get_ai_financial_insights(
                st.session_state.monthly_salary,
                estimated_monthly_tax, # Pass estimated monthly tax to AI
                budget_metrics,
                st.session_state.expenses_df, # Pass the full DataFrame for broader context
                st.session_state.user_budget_categories # Pass user's budget categories to AI
            )
            # Display the AI-generated insights (using markdown for formatting)
            st.markdown(ai_insights)

    # --- Detailed Expenses Table ---
    st.subheader("📑 Expense Details")
    # Display the raw expense data in a sortable table
    st.dataframe(
        st.session_state.expenses_df.sort_values('Date', ascending=False), # Sort by date descending
        use_container_width=True, # Make table responsive
        hide_index=True # Hide the DataFrame index
    )
