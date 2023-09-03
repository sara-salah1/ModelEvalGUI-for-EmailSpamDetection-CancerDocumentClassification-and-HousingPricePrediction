import sys
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtGui import QIcon, QGuiApplication
from PyQt5.QtWidgets import QWidget, QLabel, QApplication, QVBoxLayout, QHBoxLayout, QMainWindow, QRadioButton, \
    QComboBox, QButtonGroup, QCheckBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import mean_squared_error
import matplotlib.colors as mcolors
from CustomButton import RoundButton


class mainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.custom_cmap = None
        self.interlabel = None
        self.select_label, self.combo, self.model_label = None, None, None
        self.buttonGroup, self.check_knn, self.check_kmean, self.check_tree = None, None, None, None
        self.check_regression, self.check_naive_bayes, self.plot_button = None, None, None
        self.evaluation_label, self.ButtonGroup2 = None, None
        self.radio_accuracy, self.confusion_matrix, self.figure, self.canvas = None, None, None, None
        self.knn_model, self.DT_model, self.naive_model, self.LR_model, self.Kmean_model = None, None, None, None, None
        self.x_test, self.y_test, self.x_testCDC, self.y_testCDC = None, None, None, None
        self.x_testHPP, self.y_testHPP = None, None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("ModelEval Express")
        self.setGeometry(200, 200, 800, 800)
        self.setStyleSheet("background-color: #004b49;")
        self.custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom_colormap', ['#40826d', 'white', '#004b49'])
        self.set_icon()
        self.center()
        self.create_widgets()
        self.widget_style()
        self.widget_actions()
        self.create_layout()

    def set_icon(self):
        app_icon = QIcon("machine.png")
        self.setWindowIcon(app_icon)

    def center(self):
        q_rect = self.frameGeometry()
        center_point = QGuiApplication.primaryScreen().availableGeometry().center()
        q_rect.moveCenter(center_point)
        self.move(q_rect.topLeft())

    def create_widgets(self):
        self.interlabel = QLabel(
            "<html><div style='text-align: center;'>"
            "<p>Email Spam Detection, Cancer Document classification</p>"
            "<p>and Housing Price Prediction</p>"
            "<p>                               </p>"
            "</div></html>"
        )

        self.select_label = QLabel("Choose your dataset:")

        self.combo = QComboBox()
        self.combo.addItems(['EmailSpam.csv', 'CancerDocument.csv', 'HousePrice.csv'])
        self.combo.setCurrentIndex(-1)

        self.model_label = QLabel("Choose your models")
        self.buttonGroup = QButtonGroup()

        self.check_knn = QCheckBox('KNN', self)
        self.check_tree = QCheckBox('Decision Tree', self)
        self.check_regression = QCheckBox('Linear Regression', self)
        self.check_kmean = QCheckBox('K Mean', self)
        self.check_naive_bayes = QCheckBox('Naive Payes', self)

        self.plot_button = RoundButton("plot")

        self.evaluation_label = QLabel("Choose your evaluation method")
        self.ButtonGroup2 = QButtonGroup()

        self.radio_accuracy = QRadioButton("Accuracy", self)
        self.confusion_matrix = QRadioButton("Confusion Matrix", self)

        self.ButtonGroup2.addButton(self.radio_accuracy)
        self.ButtonGroup2.addButton(self.confusion_matrix)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)

    def widget_actions(self):
        self.combo.currentIndexChanged.connect(self.edit_widgets)

        self.plot_button.clicked.connect(self.display_plot)

        self.combo.currentIndexChanged.connect(self.enable_button)
        self.check_knn.stateChanged.connect(self.enable_button)
        self.check_tree.stateChanged.connect(self.enable_button)
        self.check_regression.stateChanged.connect(self.enable_button)
        self.check_kmean.stateChanged.connect(self.enable_button)
        self.check_naive_bayes.stateChanged.connect(self.enable_button)
        self.radio_accuracy.toggled.connect(self.enable_button)
        self.confusion_matrix.toggled.connect(self.enable_button)

    def widget_style(self):
        self.interlabel.setFont(QFont("Arial", 15, QFont.Bold))
        self.interlabel.setStyleSheet("color: #ffffff;")
        self.interlabel.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.select_label.setStyleSheet("color: #ffffff;")
        self.select_label.setFont(QFont("Arial", 13, QFont.Bold))

        self.combo.setFont(QFont("Arial", 10))
        self.combo.setStyleSheet("QComboBox { height: 50px; color: white; background-color: #40826d; "
                                 "selection-background-color: transparent;}")
        self.combo.setFixedWidth(550)

        self.model_label.setStyleSheet("color: #ffffff;")
        self.model_label.setFont(QFont("Arial", 13, QFont.Bold))

        self.model_label.setVisible(False)
        self.check_knn.setVisible(False)
        self.check_tree.setVisible(False)
        self.check_regression.setVisible(False)
        self.check_kmean.setVisible(False)
        self.check_naive_bayes.setVisible(False)

        self.plot_button.setFixedWidth(250)
        self.plot_button.setFixedHeight(40)
        self.plot_button.setDisabled(True)

        self.evaluation_label.setStyleSheet("color: #ffffff;")
        self.evaluation_label.setFont(QFont("Arial", 13, QFont.Bold))

        self.check_knn.setStyleSheet("color: white; font-size: 15px;")
        self.check_tree.setStyleSheet("color: white; font-size: 15px;")
        self.check_regression.setStyleSheet("color: white; font-size: 15px;")
        self.check_kmean.setStyleSheet("color: white; font-size: 15px;")
        self.check_naive_bayes.setStyleSheet("color: white; font-size: 15px;")

        self.radio_accuracy.setStyleSheet("color: white; font-size: 15px;")
        self.confusion_matrix.setStyleSheet("color: white; font-size: 15px;")

    def edit_widgets(self):
        selected_dataset = self.combo.currentText()
        self.model_label.setVisible(True)

        if selected_dataset == 'EmailSpam.csv':
            self.check_knn.setVisible(True)
            self.check_tree.setVisible(True)
            self.check_naive_bayes.setVisible(True)
            self.check_regression.setVisible(False)
            self.check_kmean.setVisible(False)
            self.radio_accuracy.setVisible(True)
            self.confusion_matrix.setVisible(True)
        elif selected_dataset == 'CancerDocument.csv':
            self.check_knn.setVisible(False)
            self.check_tree.setVisible(False)
            self.check_naive_bayes.setVisible(False)
            self.check_regression.setVisible(False)
            self.check_kmean.setVisible(True)
            self.radio_accuracy.setVisible(False)
            self.confusion_matrix.setVisible(True)
        elif selected_dataset == 'HousePrice.csv':
            self.check_knn.setVisible(False)
            self.check_tree.setVisible(False)
            self.check_regression.setVisible(True)
            self.check_kmean.setVisible(False)
            self.check_naive_bayes.setVisible(False)
            self.radio_accuracy.setVisible(True)
            self.confusion_matrix.setVisible(False)

    def create_layout(self):
        page_layout = QVBoxLayout()
        hbox = QHBoxLayout()
        vbox_model = QVBoxLayout()
        vbox_evaluation = QVBoxLayout()

        hbox.addWidget(self.select_label)
        hbox.addWidget(self.combo)

        vbox_model.addWidget(self.model_label)
        vbox_model.addWidget(self.check_knn)
        vbox_model.addWidget(self.check_tree)
        vbox_model.addWidget(self.check_regression)
        vbox_model.addWidget(self.check_kmean)
        vbox_model.addWidget(self.check_naive_bayes)

        vbox_evaluation.addWidget(self.evaluation_label)
        vbox_evaluation.addWidget(self.radio_accuracy)
        vbox_evaluation.addWidget(self.confusion_matrix)

        hbox_center = QHBoxLayout()
        hbox_center.addStretch()
        hbox_center.addWidget(self.plot_button)
        hbox_center.addStretch()

        page_layout.addWidget(self.interlabel)
        page_layout.addLayout(hbox)
        page_layout.addLayout(vbox_model)
        page_layout.addLayout(vbox_evaluation)
        page_layout.addLayout(hbox_center)
        page_layout.addWidget(self.canvas)

        widget = QWidget()
        widget.setLayout(page_layout)
        self.setCentralWidget(widget)

    def enable_button(self):
        dataset_selected = self.combo.currentText()
        model_selected = any(
            [self.check_knn.isChecked(), self.check_tree.isChecked(), self.check_regression.isChecked(),
             self.check_kmean.isChecked(), self.check_naive_bayes.isChecked()])
        evaluation_selected = any([self.radio_accuracy.isChecked(), self.confusion_matrix.isChecked()])

        if dataset_selected and model_selected and evaluation_selected:
            self.plot_button.setEnabled(True)
        else:
            self.plot_button.setDisabled(True)

    def get_modelsESD(self):
        self.knn_model = joblib.load("C:/Users/LEGION/PycharmProjects/GradProjectITI/knn_model.pk1")
        self.DT_model = joblib.load("C:/Users/LEGION/PycharmProjects/GradProjectITI/DT_model.pk1")
        self.naive_model = joblib.load("C:/Users/LEGION/PycharmProjects/GradProjectITI/naive_model.pk1")

    def read_test_dataESD(self):
        self.x_test = pd.read_csv("C:/Users/LEGION/PycharmProjects/GradProjectITI/models/x_test.csv")
        self.y_test = pd.read_csv("C:/Users/LEGION/PycharmProjects/GradProjectITI/models/y_test.csv")

    def get_modelsHPP(self):
        self.LR_model = joblib.load("C:/Users/LEGION/PycharmProjects/GradProjectITI/LR_model.pk1")

    def read_test_dataHPP(self):
        self.x_testHPP = pd.read_csv("C:/Users/LEGION/PycharmProjects/GradProjectITI/models/x_testHPP.csv")
        self.y_testHPP = pd.read_csv("C:/Users/LEGION/PycharmProjects/GradProjectITI/models/y_testHPP.csv")

    def get_modelsCDC(self):
        self.Kmean_model = joblib.load("C:/Users/LEGION/PycharmProjects/GradProjectITI/Kmean_model.pk1")

    def read_test_dataCDC(self):
        self.x_testCDC = pd.read_csv("C:/Users/LEGION/PycharmProjects/GradProjectITI/models/x_testCDC.csv")
        self.y_testCDC = pd.read_csv("C:/Users/LEGION/PycharmProjects/GradProjectITI/models/y_testCDC.csv")

    def get_test_accuracyESD(self):
        y_pred_knn = self.knn_model.predict(self.x_test)
        y_pred_dt = self.DT_model.predict(self.x_test)
        y_pred_nb = self.naive_model.predict(self.x_test)

        accuracy_knn = accuracy_score(self.y_test, y_pred_knn)
        accuracy_dt = accuracy_score(self.y_test, y_pred_dt)
        accuracy_nb = accuracy_score(self.y_test, y_pred_nb)

        return {
            'KNN': accuracy_knn,
            'Decision Tree': accuracy_dt,
            'Naive Payes': accuracy_nb,
        }

    def get_test_confusion_matrix_ESD(self):
        confusion_matrix_data = {}
        for model_name in self.models_selected:
            if model_name == 'KNN':
                y_pred_knn = self.knn_model.predict(self.x_test)
                confusion_matrix_data['KNN'] = confusion_matrix(self.y_test, y_pred_knn)
            elif model_name == 'Decision Tree':
                y_pred_dt = self.DT_model.predict(self.x_test)
                confusion_matrix_data['Decision Tree'] = confusion_matrix(self.y_test, y_pred_dt)
            elif model_name == 'Naive Payes':
                y_pred_nb = self.naive_model.predict(self.x_test)
                confusion_matrix_data['Naive Payes'] = confusion_matrix(self.y_test, y_pred_nb)
        return confusion_matrix_data

    def get_test_accuracyHPP(self):
        y_pred_LR = self.LR_model.predict(self.x_testHPP)
        rmse = np.sqrt(mean_squared_error(self.y_testHPP, y_pred_LR))

        return {
            'Linear Regression': rmse
        }

    def get_test_confusion_matrix_CDC(self):
        y_pred_kmean = self.Kmean_model.predict(self.x_testCDC)
        cm_kmean = confusion_matrix(self.y_testCDC, y_pred_kmean)

        return {
            'K Mean': cm_kmean
        }

    def display_plot(self):
        self.dataset_selected = self.combo.currentText()
        self.models_selected = [model.text() for model in
                                [self.check_knn, self.check_tree, self.check_regression, self.check_kmean,
                                 self.check_naive_bayes] if model.isChecked()]
        self.evaluation_method = 'Accuracy' if self.radio_accuracy.isChecked() else 'Confusion Matrix'

        try:
            if self.dataset_selected == 'EmailSpam.csv':
                self.get_modelsESD()
                self.read_test_dataESD()
                if self.evaluation_method == 'Accuracy':
                    self.accuracy_data = self.get_test_accuracyESD()
                    self.plot_accuracyESD(self.accuracy_data, self.models_selected)
                elif self.evaluation_method == 'Confusion Matrix':
                    self.confusion_matrix_data = self.get_test_confusion_matrix_ESD()
                    self.plot_confusion_matrix_ESD(self.confusion_matrix_data, self.models_selected)

            elif self.dataset_selected == 'HousePrice.csv':
                self.get_modelsHPP()
                self.read_test_dataHPP()
                if self.evaluation_method == 'Accuracy':
                    self.accuracy_data = self.get_test_accuracyHPP()
                    self.plot_accuracyHPP(self.accuracy_data, self.models_selected)

            elif self.dataset_selected == 'CancerDocument.csv':
                self.get_modelsCDC()
                self.read_test_dataCDC()
                if self.evaluation_method == 'Confusion Matrix':
                    self.confusion_matrix_data = self.get_test_confusion_matrix_CDC()
                    self.plot_confusion_matrix_CDC(self.confusion_matrix_data, self.models_selected)

        except Exception as e:
            print(f'An error occurred: {e}')

        print('Reached the end of display_plot function.')

    def plot_accuracyESD(self, accuracy_data, selected_models):
        model_names = []
        accuracies = []

        for model_name in selected_models:
            accuracy = accuracy_data.get(model_name, None)
            if accuracy is not None:
                model_names.append(model_name)
                accuracies.append(accuracy)

        self.plot_line_graphESD(model_names, accuracies)

    def plot_accuracyHPP(self, accuracy_data, selected_models):
        model_names = []
        accuracies = []

        for model_name in selected_models:
            accuracy = accuracy_data.get(model_name, None)
            if accuracy is not None:
                model_names.append(model_name)
                accuracies.append(accuracy)

        self.plot_line_graphHPP(model_names, accuracies)

    def plot_confusion_matrix_ESD(self, confusion_matrix_data, selected_models):
        num_models = len(selected_models)
        if num_models > 1:
            num_rows = num_models // 2 + num_models % 2
            num_cols = 2
            axes = []
            self.figure.clear()
            for i, model_name in enumerate(selected_models):
                confusion_matrix = confusion_matrix_data.get(model_name, None)
                if isinstance(confusion_matrix, (pd.DataFrame, np.ndarray)) and confusion_matrix.ndim == 2:

                    row_idx = i // 3
                    col_idx = i % 2
                    ax = self.figure.add_subplot(num_rows, num_cols, i + 1)
                    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap=self.custom_cmap, cbar=False, ax=ax)
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('True')
                    ax.set_title(f'Confusion Matrix - {model_name}')
                    axes.append(ax)
                    self.figure.tight_layout()
                    self.figure.canvas.draw()

            for i in range(num_models, num_rows * num_cols):
                ax = self.figure.add_subplot(num_rows, num_cols, i + 1)
                ax.axis('off')

        else:
            model_name = selected_models[0]
            confusion_matrix = confusion_matrix_data.get(model_name, None)
            if isinstance(confusion_matrix, (pd.DataFrame, np.ndarray)) and confusion_matrix.ndim == 2:
                self.figure.clear()
                ax = self.figure.add_subplot(111)
                sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap=self.custom_cmap, cbar=False, ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('True')
                ax.set_title(f'Confusion Matrix - {model_name}')

        self.figure.tight_layout()
        self.figure.canvas.draw()

    def plot_confusion_matrix_CDC(self, confusion_matrix_data, selected_models):
        for model_name in selected_models:
            confusion_matrix = confusion_matrix_data.get(model_name, None)
            if isinstance(confusion_matrix, (pd.DataFrame, np.ndarray)) and confusion_matrix.ndim == 2:
                self.figure.clear()
                ax = self.figure.add_subplot(111)
                sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap=self.custom_cmap, cbar=False, ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('True')
                ax.set_title(f'Confusion Matrix - {model_name}')
                self.figure.tight_layout()
                self.figure.canvas.draw()

    def plot_line_graphESD(self, model_names, accuracies):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        bars = ax.bar(model_names, accuracies, color='#40826d')

        ax.set_xlabel('Model')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy for Selected Models')

        for bar, accuracy in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f'{accuracy:.2f}',
                    ha='center', va='bottom')

        self.figure.tight_layout()
        self.canvas.draw()

    def plot_line_graphHPP(self, model_names, rmse_values):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        bars = ax.bar(model_names, rmse_values, color='#40826d')

        ax.set_xlabel('Model')
        ax.set_ylabel('RMSE')
        ax.set_title('Root Mean Square Error (RMSE) for Selected Models')

        for bar, rmse in zip(bars, rmse_values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f'{rmse * 100:.2f}%',
                    ha='center', va='bottom')

        self.figure.tight_layout()
        self.canvas.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = mainWindow()
    main_window.show()
    app.exec()
