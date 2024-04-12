from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any, List
import string
import jsonpickle
import json
import numpy as np


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders),
            conversions,
            "",
            "",
        ]))

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(self.to_json([
            self.compress_state(state, self.truncate(
                state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append(
                [listing["symbol"], listing["product"], listing["denomination"]])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [
                order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append([
                    trade.symbol,
                    trade.price,
                    trade.quantity,
                    trade.buyer,
                    trade.seller,
                    trade.timestamp,
                ])

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sunlight,
                observation.humidity,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[:max_length - 3] + "..."


logger = Logger()


class Trader:
    class SimpleLinearRegression:
        def __init__(self):
            self.slope = None
            self.intercept = None

        def fit(self, X: List[float], y: List[float]) -> None:
            n = len(X)
            if n != len(y):
                raise ValueError("X and y must have the same length")

            mean_X = np.mean(X)
            mean_y = np.mean(y)

            # Compute the slope (beta1) and intercept (beta0)
            numerator = sum((X[i] - mean_X) * (y[i] - mean_y)
                            for i in range(n))
            denominator = sum((X[i] - mean_X)**2 for i in range(n))

            self.slope = numerator / denominator
            self.intercept = mean_y - self.slope * mean_X

        def predictLR(self, X: List[float]) -> List[float]:
            if self.slope is None or self.intercept is None:
                raise ValueError(
                    "The model has not been trained yet. Call fit() first.")

            return [self.intercept + self.slope * x for x in X]

        def predict_one_std_above(self, X: List[float], y_std: float) -> List[float]:
            if self.slope is None or self.intercept is None:
                raise ValueError(
                    "The model has not been trained yet. Call fit() first.")

            predictions = self.predictLR(X)
            return [pred + y_std for pred in predictions]

        def predict_one_std_below(self, X: List[float], y_std: float) -> List[float]:
            if self.slope is None or self.intercept is None:
                raise ValueError(
                    "The model has not been trained yet. Call fit() first.")

            predictions = self.predictLR(X)
            return [pred - y_std for pred in predictions]

    def to_json(self) -> str:
        return jsonpickle.encode(self)

    @staticmethod
    def from_json(json_str: str) -> 'SimpleLinearRegression':
        return jsonpickle.decode(json_str)

    class TraderData:
        def __init__(self):
            self.round = 0
            self.mean = 0
            self.ema = 0
            self.avg_prices = []  # length 10 array
            self.ema_prices = []
            # length 20 array of tuples: (x, y) -> (round, avg_price)
            self.lr_prices = []
            self.lr: Trader.SimpleLinearRegression = Trader.SimpleLinearRegression()

        def toJsonStr(self):
            return f"{jsonpickle.encode(self)}"

        def updateMean(self, new_val: float):
            # simple rolling mean
            # if self.round == 0:
            #     self.mean = new_val
            # else:
            #     self.mean = ((self.mean * self.round) +
            #                  new_val) / (self.round + 1)

            self.avg_prices.append(new_val)
            if len(self.avg_prices) > 20:
                self.avg_prices = self.avg_prices[1:]
            self.mean = sum(self.avg_prices) / len(self.avg_prices)

        def updateEMA(self, new_val: float):
            self.ema_prices.append(new_val)
            if len(self.ema_prices) > 10:
                self.ema_prices = self.ema_prices[1:]

            self.ema = self.ema_prices[0]
            for i in range(1, len(self.ema_prices)):
                self.ema = (
                    self.ema_prices[i] * (2 / (2 + i))) + self.ema * (1 - (2 / (2 + i)))

        def updateLR(self, new_val: float):
            self.lr_prices.append((self.round, new_val))  # round is timestamp
            if len(self.lr_prices) > 10:
                self.lr_prices = self.lr_prices[1:]

            self.lr.fit([p[0] for p in self.lr_prices], [p[1]
                        for p in self.lr_prices])
            logger.print('lin reg slope:', self.lr.slope)

        @staticmethod
        def fromJsonStr(json_str: str):
            if not json_str:
                return Trader.TraderData()

            obj = jsonpickle.decode(json_str)
            new = Trader.TraderData()

            # Use getattr to safely retrieve attributes from obj
            for attr in ['round', 'mean', 'ema', 'avg_prices', 'ema_prices', 'lr_prices', 'lr']:
                if hasattr(obj, attr):
                    setattr(new, attr, getattr(obj, attr))

            return new

    def __get_avg_price(self, order_depth):
        # get average price from all buy and sell orders for a single OrderDepth object
        (s_prices, b_prices) = (
            order_depth.sell_orders.keys(), order_depth.buy_orders.keys())
        return (sum(s_prices) + sum(b_prices)) / (len(s_prices) + len(b_prices))

    def __get_avg_price_market(self, market_trades):
        print()

    def run(self, state: TradingState):
        # Only method required. It takes all buy and sell orders for all symbols as an input, and outputs a list of orders to be sent
        data: Trader.TraderData = Trader.TraderData.fromJsonStr(
            state.traderData)

        result = {}
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            if product in state.position:
                apos = state.position[product]
            else:
                apos = 0
            print(f"position for {product}: {apos}")
            orders: List[Order] = []
            sell_qty = 0
            buy_qty = 0

            avg_price = self.__get_avg_price(order_depth)

            if product == "AMETHYSTS":
                # go through outstanding buy orders, then sell it to the ones with the highest prices
                for ask_price, quantity in sorted(order_depth.buy_orders.items(), reverse=True):
                    if ask_price > 10000 or (ask_price == 10000 and apos > 0):
                        # logger.print(sell_qty, apos, quantity)
                        qty = min(max(0, 20 -
                                      (sell_qty - apos)), quantity)
                        sell_qty += qty
                        orders.extend([Order("AMETHYSTS", ask_price, -1)
                                      for _ in range(qty)])

                        apos -= qty

                # go through all outstanding sell orders, and buy the best ones
                # sell orders: { price : quantity }
                for sell_price, val in sorted(order_depth.sell_orders.items()):
                    quantity = abs(val)
                    if sell_price < 10000 or (sell_price == 10000 and apos < 0):
                        qty = min(max(0, 20-(buy_qty+apos)), quantity)
                        buy_qty += qty
                        orders.extend([Order("AMETHYSTS", sell_price, 1)
                                      for _ in range(qty)])

                        apos += qty

                result[product] = orders

            elif product == "STARFRUIT":
                stddev = 0.5 * np.std([p[1] for p in data.lr_prices])
                if data.round > 0:
                    prediction_below = data.lr.predict_one_std_below([data.round], stddev)[
                        0]
                    prediction_above = data.lr.predict_one_std_above([data.round], stddev)[
                        0]
                    prediction = data.lr.predictLR([data.round])[0]
                else:
                    prediction_above = avg_price
                    prediction_below = avg_price
                    prediction = avg_price

                logger.print(prediction_below, prediction_above)

                # stop loss
                # if apos > 0 and avg_price < prediction_below - 0.01 * stddev:
                #     orders.append(Order("STARFRUIT", round(avg_price), -apos))
                # elif apos < 0 and avg_price < prediction_above + 0.01 * stddev:
                #     orders.append(Order("STARFRUIT", round(avg_price), -apos))

                # go through all outstanding sell orders, and buy the best ones
                # sell orders: { price : quantity }
                for sell_price, val in sorted(order_depth.sell_orders.items()):
                    quantity = abs(val)
                    if sell_price <= prediction_below or (sell_price >= prediction_below and sell_price <= prediction and apos < 0):
                        qty = min(max(0, 20-(buy_qty+apos)), quantity)
                        buy_qty += qty
                        orders.extend([Order("STARFRUIT", sell_price, 1)
                                      for _ in range(qty)])

                        apos += qty

                # go through outstanding buy orders, then sell it to the ones with the highest prices
                for ask_price, quantity in sorted(order_depth.buy_orders.items(), reverse=True):
                    if ask_price >= prediction_above or (ask_price <= prediction_above and ask_price >= prediction and apos > 0):
                        qty = min(max(0, 20 - (sell_qty - apos)), quantity)
                        sell_qty += qty
                        orders.extend([Order("STARFRUIT", ask_price, -1)
                                      for _ in range(qty)])

                        apos -= qty

                result[product] = orders

                data.updateEMA(avg_price)
                data.updateLR(avg_price)
                logger.print("AVERAGE PRICE:", avg_price)
                data.round += 1

            # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.

            traderData = data.toJsonStr()
            conversions = None

        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData
