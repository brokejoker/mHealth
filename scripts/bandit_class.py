# import matplotlib.pyplot as plt
# import json
import pandas as pd
import numpy as np
import matplotlib as mpl
# import matplotlib
# mpl.use('TkAgg')
mpl.use('module://kivy.garden.matplotlib.backend_kivy')
import matplotlib.pyplot as plt
# mpl.use("MacOSX")
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', 0)
pd.set_option('display.max_columns', None)
from scipy.stats import norm

from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvas

import seaborn
# plt.style.use(['seaborn-poster'])
from kivy.uix.boxlayout import BoxLayout
# import tkinter
# root = tkinter.Tk()
# root.withdraw()

from IPython.core.display import display, HTML
# display(HTML("<style>.container { width:100% !important; }</style>"))


class SimulatedContextualBandit(GridLayout):

    def __init__(self, num_context_feats=7, num_actions=2, num_sbar_feats=4,
                 pi_min=0.2, pi_max=0.8, num_timesteps=150, ro=0.1,
                 const_theta=[0.116, -0.275, -0.233, 0.425, 0.116, 0.275,
                              -0.233, 0.0425], reward_thresh=0.8):
        self.num_context_feats = num_context_feats
        self.num_actions = num_actions
        self.num_sbar_feats = num_sbar_feats
        self.pi_min = pi_min
        self.pi_max = pi_max
        self.num_timesteps = num_timesteps
        self.ro = ro
        self.CONST_THETA = const_theta
        self.reward_thresh=0.8
        self.s_bar_dim = self.num_actions*self.num_context_feats

        self.B = np.identity(num_sbar_feats)
        self.theta_hat = np.zeros(num_sbar_feats)
        self.b_hat = np.zeros(num_sbar_feats)

        self.pi = [0]*num_timesteps
        self.pi[0] = np.random.uniform(low=pi_min, high=pi_max)
        self.mean = np.zeros(num_context_feats)
        self.cov = np.identity(num_context_feats)

        self.s_bar = [0] * num_timesteps
        self.s_bar[0] = np.ones((num_context_feats, 1))

        self.b_hat_ls = []
        self.theta_ls = []

        self.n = [0] * num_timesteps
        self.ns = [0] * num_timesteps
        self.a = [0] * num_timesteps
        self.s_ta_bar = [0] * num_timesteps
        self.reward = [0] * num_timesteps

        self.s_ta = [0] * num_timesteps
        self.eta_bar = [0] * num_timesteps
        self.nu = 1
        self.s_ls = []
        self.b_ls = []

    def run_sumulation(self):
        for t in range(1, self.num_timesteps):
            # sample n[t]
            self.n[t] = np.random.multivariate_normal(self.mean, self.cov, 1)
            self.ns[t] = np.random.multivariate_normal(self.mean, self.cov, 1)

            # update s_bar (i.e "observe" s_bar)
            self.s_bar[t] = (np.sqrt(1 - self.ro ** 2) * self.s_bar[t - 1] +
                             self.ro * self.ns[t].reshape(-1, 1))

            # Make s_ta
            self.s_ta[t] = np.array(
                [self.s_bar[t][:self.num_sbar_feats] for i in
                 range(self.num_actions)])

            # update eta_bar
            self.eta_bar[t] = (np.sqrt(1 - self.ro ** 2) * self.eta_bar[t - 1]
                              + self.ro * self.n[t].reshape(-1, 1))

            # Sample theta_prime
            theta_prime = np.random.multivariate_normal(self.theta_hat,
                                                        self.nu ** 2 *
                                                        np.linalg.inv(self.B), 1)

            # Which action is optimal?
            opt_action_idx = np.argmax(
                (self.s_ta[t] * theta_prime.reshape(-1, 1)).sum(axis=1))

            # Prob of taking non-zero action
            cdf = norm.cdf(0,
                           loc=sum((self.s_bar[t][:self.num_sbar_feats] *
                                    self.theta_hat.reshape(-1, 1))),
                           scale=sum((self.s_bar[t][:self.num_sbar_feats] *
                                      self.nu ** 2 * np.diagonal(
                                      np.linalg.inv(self.B)).reshape(-1, 1))))
            self.pi[t] = float(max(self.pi_min, min(self.pi_max, cdf)))

            # Flip biased coin with prob pi[t]
            outcome = np.random.binomial(1, self.pi[t])

            if outcome:
                self.a[t] = opt_action_idx
                a_bar_indicator = 1
            else:
                self.a[t] = 0
                a_bar_indicator = 0
            # Optimal non-zero action
            self.s_ta_bar[t] = self.s_ta[t][opt_action_idx]

            s = np.linalg.norm(self.s_bar[t], 1)
            self.s_ls.append(s)
            if s > 0.8:
                state_indicator = 1
            else:
                state_indicator = 0

            # Simulated reward
            self.reward[t] = sum(
                np.array(self.CONST_THETA).reshape(1, -1) * np.concatenate(
                    self.s_ta[t]).reshape(1, -1))[
                            0] + 2 * state_indicator + np.random.normal(0, 1)
            # Update b_hat, B, and theta_hat
            self.B = (self.B + self.pi[t] * (1 - self.pi[t]) * self.s_ta_bar[t] *
                 self.s_ta_bar[t])
            self.b_hat = (self.b_hat + np.concatenate(self.s_ta_bar[t] *
                            (a_bar_indicator - self.pi[t]) * self.reward[t]))
            self.b_ls.append(self.b_hat)
            self.theta_hat = np.matmul(np.linalg.inv(self.B), self.b_hat).ravel()
            self.theta_ls.append(self.theta_hat)

        # print("theta_hat:".format(self.theta_hat))
        fig1 = plt.plot([np.linalg.norm(self.theta_ls[t] - self.theta_ls[t - 1],
                                 1) / np.linalg.norm(self.theta_ls[t], 1) for t
                  in range(1, self.num_timesteps - 1)], label='theta_hat')
        plt.title(
            'Norm of difference of adjacent theta_hat iterations by timestep t')
        plt.xlabel('t')
        plt.ylabel('proportional difference of norms')

        fig2 = plt.plot([np.linalg.norm(self.b_ls[t] - self.b_ls[t - 1],
                                 1) / np.linalg.norm(self.b_ls[t], 1) for t in
                  range(1, self.num_timesteps - 1)], label='b_hat')
        plt.title(
            'Norm of difference of adjacent theta_hat & b_hat iterations by timestep t')
        plt.xlabel('t')
        plt.ylabel('proportional difference of norms')

        plt.legend()
        # plt.figure()
        fig3 = plt.plot(self.pi, label='b_hat')
        plt.title('pi[t]')
        plt.xlabel('t')
        plt.ylabel('proportional difference of norms')


        box = BoxLayout()
        box.add_widget(FigureCanvas(plt.gcf()))
        # box.add_widget(FigureCanvas(plt.gcf()))
        # box.add_widget(FigureCanvas(plt.gcf()))




        return box


class MyApp(App):
    def build(self):
        return SimulatedContextualBandit().run_sumulation()

if __name__ == '__main__':
    MyApp().run()