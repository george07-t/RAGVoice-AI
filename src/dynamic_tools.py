"""
Dynamic Function Calling Tools for Voice AI Agent
==================================================

This module provides external API integration for:
- Weather information (WeatherAPI.com)
- Exchange rates (Exchangerate-API)
- Additional utility functions

All tools follow LangChain's @tool decorator pattern for seamless integration.
"""

import logging
import os

import requests
from dotenv import load_dotenv
from langchain_core.tools import tool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(".env.local")

# API Configuration
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "32ad9a51a0874338bf541757252904")
EXCHANGE_RATE_API_KEY = os.getenv("EXCHANGE_RATE_API_KEY")  # Optional


@tool
def get_current_weather(city: str = "Dhaka") -> str:
    """
    Get current weather information for any city worldwide.

    Provides real-time weather data including temperature, conditions,
    humidity, wind speed, and feels-like temperature.

    Args:
        city: Name of the city to get weather for. Defaults to "Dhaka"
    """
    try:
        logger.info(f"Fetching weather for {city}")

        # WeatherAPI.com endpoint
        url = "http://api.weatherapi.com/v1/current.json"
        params = {
            "key": WEATHER_API_KEY,
            "q": city,
            "aqi": "no"  # Air quality index not needed for basic queries
        }

        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()

        data = response.json()

        # Extract relevant information
        location = data["location"]
        current = data["current"]

        # Format response
        weather_info = f"""
🌍 Weather in {location['name']}, {location['country']}
🌡️ Temperature: {current['temp_c']}°C ({current['temp_f']}°F)
🌤️ Condition: {current['condition']['text']}
🤔 Feels like: {current['feelslike_c']}°C ({current['feelslike_f']}°F)
💧 Humidity: {current['humidity']}%
💨 Wind: {current['wind_kph']} km/h ({current['wind_dir']})
🕐 Last updated: {current['last_updated']}
        """.strip()

        logger.info(f"Successfully retrieved weather for {city}")
        return weather_info

    except requests.exceptions.RequestException as e:
        error_msg = f"Error fetching weather data: {e!s}"
        logger.error(error_msg)
        return f"Sorry, I couldn't retrieve weather information for {city}. Please check the city name and try again."
    except KeyError as e:
        error_msg = f"Error parsing weather data: {e!s}"
        logger.error(error_msg)
        return "Sorry, I received unexpected data format from the weather service."
    except Exception as e:
        logger.error(f"Unexpected error in get_current_weather: {e}")
        return "An unexpected error occurred while fetching weather information."


@tool
def get_weather_forecast(city: str = "Dhaka", days: int = 3) -> str:
    """
    Get weather forecast for the next few days for any city.

    Args:
        city: Name of the city to get forecast for. Defaults to "Dhaka"
        days: Number of days to forecast (1-3). Defaults to 3
    """
    try:
        logger.info(f"Fetching {days}-day forecast for {city}")

        # Limit days to API's free tier (3 days)
        days = min(max(days, 1), 3)

        url = "http://api.weatherapi.com/v1/forecast.json"
        params = {
            "key": WEATHER_API_KEY,
            "q": city,
            "days": days,
            "aqi": "no",
            "alerts": "no"
        }

        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()

        data = response.json()
        location = data["location"]
        forecast_days = data["forecast"]["forecastday"]

        # Format forecast
        forecast_text = f"📅 {days}-Day Weather Forecast for {location['name']}, {location['country']}\n\n"

        for day_data in forecast_days:
            date = day_data["date"]
            day = day_data["day"]

            forecast_text += f"""
📆 {date}
  🌡️ High: {day['maxtemp_c']}°C / Low: {day['mintemp_c']}°C
  🌤️ Condition: {day['condition']['text']}
  🌧️ Chance of rain: {day['daily_chance_of_rain']}%
  💨 Max wind: {day['maxwind_kph']} km/h
            """.strip() + "\n"

        logger.info(f"Successfully retrieved forecast for {city}")
        return forecast_text.strip()

    except Exception as e:
        logger.error(f"Error in get_weather_forecast: {e}")
        return f"Sorry, I couldn't retrieve the weather forecast for {city}."


@tool
def get_exchange_rate(base_currency: str = "USD", target_currency: str = "BDT") -> str:
    """
    Get current exchange rate between two currencies.

    Args:
        base_currency: Base currency code (e.g., "USD", "EUR"). Defaults to "USD"
        target_currency: Target currency code (e.g., "BDT", "EUR"). Defaults to "BDT"
    """
    try:
        logger.info(f"Fetching exchange rate: {base_currency} -> {target_currency}")

        # Using exchangerate-api.com (free tier: 1500 requests/month)
        url = f"https://api.exchangerate-api.com/v4/latest/{base_currency.upper()}"

        response = requests.get(url, timeout=5)
        response.raise_for_status()

        data = response.json()
        rates = data["rates"]

        target_upper = target_currency.upper()

        if target_upper not in rates:
            return f"Sorry, I couldn't find exchange rate for {target_upper}."

        rate = rates[target_upper]
        last_updated = data.get("date", "recently")

        exchange_info = f"""
💱 Exchange Rate
{base_currency.upper()} → {target_upper}
Rate: 1 {base_currency.upper()} = {rate:.4f} {target_upper}
Last updated: {last_updated}

Examples:
• 10 {base_currency.upper()} = {rate * 10:.2f} {target_upper}
• 100 {base_currency.upper()} = {rate * 100:.2f} {target_upper}
• 1000 {base_currency.upper()} = {rate * 1000:.2f} {target_upper}
        """.strip()

        logger.info("Successfully retrieved exchange rate")
        return exchange_info

    except Exception as e:
        logger.error(f"Error in get_exchange_rate: {e}")
        return f"Sorry, I couldn't retrieve the exchange rate for {base_currency} to {target_currency}."


@tool
def get_current_time(timezone: str = "Asia/Dhaka") -> str:
    """
    Get current date and time for a specific timezone.

    Args:
        timezone: Timezone name (e.g., "Asia/Dhaka", "America/New_York", "Europe/London")
    """
    try:
        from datetime import datetime

        import pytz

        tz = pytz.timezone(timezone)
        current_time = datetime.now(tz)

        time_info = f"""
🕐 Current Time
Timezone: {timezone}
Date: {current_time.strftime('%A, %B %d, %Y')}
Time: {current_time.strftime('%I:%M:%S %p')}
24-hour: {current_time.strftime('%H:%M:%S')}
        """.strip()

        return time_info

    except Exception as e:
        logger.error(f"Error in get_current_time: {e}")
        # Fallback to simple datetime if pytz not available
        now = datetime.now()
        return f"Current time (system): {now.strftime('%Y-%m-%d %H:%M:%S')}"


# Tool registry for easy access
AVAILABLE_TOOLS = [
    get_current_weather,
    get_weather_forecast,
    get_exchange_rate,
    get_current_time,
]


# Test function
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Dynamic Function Calling Tools")
    print("=" * 60)

    print("\n[Weather API Test]")
    print("-" * 60)
    result = get_current_weather.invoke({"city": "Dhaka"})
    print(result)

    print("\n\n[Weather Forecast Test]")
    print("-" * 60)
    result = get_weather_forecast.invoke({"city": "Dhaka", "days": 2})
    print(result)

    print("\n\n[Exchange Rate Test]")
    print("-" * 60)
    result = get_exchange_rate.invoke({"base_currency": "USD", "target_currency": "BDT"})
    print(result)

    print("\n\n[Current Time Test]")
    print("-" * 60)
    result = get_current_time.invoke({"timezone": "Asia/Dhaka"})
    print(result)

    print("\n" + "=" * 60)
    print("[SUCCESS] All tools tested!")
    print("=" * 60)
