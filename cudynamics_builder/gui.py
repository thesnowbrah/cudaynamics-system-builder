from __future__ import annotations

import traceback
from pathlib import Path

from PySide6.QtCore import QLocale, Qt
from PySide6.QtGui import QDoubleValidator, QFont, QIntValidator, QPixmap
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout, QFrame, QGridLayout,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow, QMessageBox,
    QPushButton, QScrollArea, QSplitter, QStackedWidget,
    QTableWidget, QTableWidgetItem, QTextEdit, QVBoxLayout, QWidget,
)

from .generator import generate
from .installer import apply_plan, build_plan, locate_default_project
from .model import AttributeDefaults, Equation, EquationKind, METHODS, SystemSpec
from .ocr import recognize_image
from .parser import ParseError, parse_system, pretty_equations, safe_identifier, sympify_expression, validate_spec


STYLE = """
QWidget { background: #101820; color: #d8e5e9; font-family: "Segoe UI"; font-size: 10pt; }
QMainWindow { background: #0c141b; }
QFrame#sidebar { background: #0b222a; border-right: 1px solid #1f4d59; }
QLabel#brand { color: #57d7dd; font-size: 18pt; font-weight: 700; padding: 18px 12px; }
QLabel#muted { color: #86a1aa; }
QLabel#section { color: #75e0df; font-size: 15pt; font-weight: 600; }
QPushButton { background: #173742; border: 1px solid #286574; border-radius: 4px; padding: 8px 12px; }
QPushButton:hover { background: #20505d; border-color: #48bec5; }
QPushButton:pressed { background: #12303a; }
QPushButton#primary { background: #168b95; border-color: #54dde0; color: white; font-weight: 600; }
QPushButton#nav { text-align: left; border: 0; border-left: 3px solid transparent; background: transparent; padding: 11px 14px; }
QPushButton#nav:checked { background: #163943; border-left-color: #54d8dc; color: #71e3e4; }
QLineEdit, QTextEdit, QComboBox, QTableWidget {
    background: #0b1319; border: 1px solid #294751; border-radius: 3px; selection-background-color: #197f89;
}
QLineEdit, QComboBox { min-height: 28px; padding: 2px 6px; }
QHeaderView::section { background: #17333d; color: #b8d5da; border: 0; border-right: 1px solid #294751; padding: 7px; }
QGroupBox { border: 1px solid #294751; border-radius: 5px; margin-top: 12px; padding-top: 12px; }
QGroupBox::title { color: #63d7d8; subcontrol-origin: margin; left: 10px; padding: 0 5px; }
QScrollBar:vertical { background: #0c151b; width: 10px; }
QScrollBar::handle:vertical { background: #315762; min-height: 24px; border-radius: 4px; }
"""


class NoWheelComboBox(QComboBox):
    """Combo box whose value cannot be changed accidentally by wheel hover."""

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt virtual method name
        event.ignore()


class MainWindow(QMainWindow):
    def __init__(self, root: Path):
        super().__init__()
        self.root = root
        self.spec: SystemSpec | None = None
        self.image_path = ""
        self.plan = None
        self.setWindowTitle("CUDAynamics · System Builder")
        self.resize(1320, 840)
        self.setStyleSheet(STYLE)
        self._build_ui()
        self._set_page(0)

    def _build_ui(self) -> None:
        central = QWidget()
        outer = QHBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        sidebar = QFrame(objectName="sidebar")
        sidebar.setFixedWidth(245)
        side = QVBoxLayout(sidebar)
        brand = QLabel("CUDAynamics\nSystem Builder", objectName="brand")
        side.addWidget(brand)
        self.nav = []
        for index, title in enumerate(("1  Ввод и OCR", "2  Проверка модели", "3  Методы и значения", "4  Добавление")):
            button = QPushButton(title, objectName="nav", checkable=True)
            button.clicked.connect(lambda _=False, i=index: self._set_page(i))
            side.addWidget(button)
            self.nav.append(button)
        side.addStretch()
        note = QLabel("Изменения CUDAynamics выполняются\nтолько на последнем шаге.", objectName="muted")
        note.setWordWrap(True)
        side.addWidget(note)
        outer.addWidget(sidebar)

        self.pages = QStackedWidget()
        self.pages.addWidget(self._input_page())
        self.pages.addWidget(self._model_page())
        self.pages.addWidget(self._settings_page())
        self.pages.addWidget(self._install_page())
        outer.addWidget(self.pages, 1)
        self.setCentralWidget(central)
        self.statusBar().showMessage("Готово")

    def _page_shell(self, title: str, subtitle: str) -> tuple[QWidget, QVBoxLayout]:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(28, 24, 28, 24)
        header = QLabel(title, objectName="section")
        layout.addWidget(header)
        hint = QLabel(subtitle, objectName="muted")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        return page, layout

    def _input_page(self) -> QWidget:
        page, layout = self._page_shell("Источник уравнений", "Загрузите изображение — это основной сценарий — либо вставьте уравнения текстом. После OCR текст всегда можно исправить.")
        splitter = QSplitter()
        left = QWidget(); left_l = QVBoxLayout(left)
        image_box = QGroupBox("Изображение")
        image_l = QVBoxLayout(image_box)
        self.image_preview = QLabel("Перетащите или выберите PNG/JPG/WebP")
        self.image_preview.setAlignment(Qt.AlignCenter)
        self.image_preview.setMinimumHeight(290)
        self.image_preview.setStyleSheet("border: 1px dashed #3b6872; background:#091116;")
        image_l.addWidget(self.image_preview)
        choose = QPushButton("Выбрать изображение…")
        choose.clicked.connect(self._choose_image)
        image_l.addWidget(choose)
        ocr_row = QHBoxLayout()
        self.ocr_backend = QComboBox(); self.ocr_backend.addItems(["pix2tex", "openai", "auto"])
        self.api_key = QLineEdit(); self.api_key.setPlaceholderText("OpenAI API key (не сохраняется)"); self.api_key.setEchoMode(QLineEdit.Password)
        self.ocr_backend.currentTextChanged.connect(self._update_api_key_visibility)
        ocr_row.addWidget(self.ocr_backend); ocr_row.addWidget(self.api_key, 1)
        self._update_api_key_visibility(self.ocr_backend.currentText())
        image_l.addLayout(ocr_row)
        recognize = QPushButton("Распознать изображение", objectName="primary")
        recognize.clicked.connect(self._recognize)
        image_l.addWidget(recognize)
        left_l.addWidget(image_box)
        splitter.addWidget(left)

        right = QWidget(); right_l = QVBoxLayout(right)
        right_l.addWidget(QLabel("Распознанный / введённый текст"))
        self.source_text = QTextEdit()
        self.source_text.setPlaceholderText("dx/dt = sigma*(y-x)\ndy/dt = x*(rho-z)-y\ndz/dt = x*y-beta*z\n\nАлгебраическая переменная: q = x+y")
        right_l.addWidget(self.source_text, 1)
        names = QFormLayout()
        self.system_id = QLineEdit("new_system")
        self.display_name = QLineEdit("New system")
        names.addRow("ID папки и ядра", self.system_id)
        names.addRow("Название в CUDAynamics", self.display_name)
        right_l.addLayout(names)
        parse = QPushButton("Разобрать и показать модель →", objectName="primary")
        parse.clicked.connect(self._parse)
        right_l.addWidget(parse)
        splitter.addWidget(right)
        splitter.setSizes([500, 650])
        layout.addWidget(splitter, 1)
        return page

    def _model_page(self) -> QWidget:
        page, layout = self._page_shell("Проверка модели", "Проверьте каждую формулу. Тип «состояние» интегрируется; «алгебраическая» пересчитывается на каждой стадии метода.")
        self.eq_table = QTableWidget(0, 3)
        self.eq_table.setHorizontalHeaderLabels(["Переменная", "Тип", "Правая часть"])
        self.eq_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.eq_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        layout.addWidget(self.eq_table, 1)
        buttons = QHBoxLayout()
        add = QPushButton("+ Уравнение"); add.clicked.connect(lambda: self._add_equation_row())
        remove = QPushButton("Удалить строку"); remove.clicked.connect(self._remove_equation_row)
        validate = QPushButton("Проверить модель", objectName="primary"); validate.clicked.connect(self._sync_model)
        buttons.addWidget(add); buttons.addWidget(remove); buttons.addStretch(); buttons.addWidget(validate)
        layout.addLayout(buttons)
        self.model_summary = QLabel("Модель ещё не разобрана", objectName="muted"); self.model_summary.setWordWrap(True)
        layout.addWidget(self.model_summary)
        return page

    def _settings_page(self) -> QWidget:
        page, layout = self._page_shell("Методы и значения по умолчанию", "Выберите создаваемые схемы. Значения попадут в .txt-описание системы и будут начальными в CUDAynamics.")
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        body = QWidget(); body_l = QVBoxLayout(body)
        method_box = QGroupBox("Численные методы"); method_grid = QGridLayout(method_box)
        self.method_checks = {}
        for i, (key, label) in enumerate(METHODS.items()):
            check = QCheckBox(label); check.setChecked(key in ("ExplicitEuler", "ExplicitRungeKutta4"))
            check.toggled.connect(self._method_warning)
            method_grid.addWidget(check, i // 2, i % 2); self.method_checks[key] = check
        body_l.addWidget(method_box)
        self.method_note = QLabel("", objectName="muted"); self.method_note.setWordWrap(True); body_l.addWidget(self.method_note)

        simulation = QGroupBox("Моделирование"); form = QFormLayout(simulation)
        self.step_value = self._number_edit("0.01", integer=False)
        self.steps_value = self._number_edit("1000", integer=True)
        self.transient_value = self._number_edit("0", integer=True)
        form.addRow("Шаг h", self.step_value); form.addRow("Число шагов", self.steps_value); form.addRow("Переходные шаги", self.transient_value)
        body_l.addWidget(simulation)

        body_l.addWidget(QLabel("Начальные значения переменных"))
        self.variable_table = QTableWidget(0, 6)
        self.variable_table.setHorizontalHeaderLabels(["Имя", "Роль", "Режим", "Минимум / значение", "Максимум", "Количество"])
        for column in (0, 1, 2):
            self.variable_table.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeToContents)
        for column in (3, 4, 5):
            self.variable_table.horizontalHeader().setSectionResizeMode(column, QHeaderView.Stretch)
        body_l.addWidget(self.variable_table)
        body_l.addWidget(QLabel("Значения параметров"))
        self.parameter_table = QTableWidget(0, 5)
        self.parameter_table.setHorizontalHeaderLabels(["Имя", "Режим", "Минимум / значение", "Максимум", "Количество"])
        for column in (0, 1):
            self.parameter_table.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeToContents)
        for column in (2, 3, 4):
            self.parameter_table.horizontalHeader().setSectionResizeMode(column, QHeaderView.Stretch)
        body_l.addWidget(self.parameter_table)
        save = QPushButton("Применить настройки и подготовить изменения →", objectName="primary"); save.clicked.connect(self._prepare)
        body_l.addWidget(save)
        scroll.setWidget(body); layout.addWidget(scroll, 1)
        return page

    @staticmethod
    def _number_edit(value: str, integer: bool) -> QLineEdit:
        """Plain validated input: unlike a spin box it ignores wheel and arrow steps."""
        edit = QLineEdit(value)
        if integer:
            validator = QIntValidator(0, 100000000, edit)
        else:
            validator = QDoubleValidator(-1e100, 1e100, 16, edit)
            validator.setNotation(QDoubleValidator.ScientificNotation)
            validator.setLocale(QLocale.c())
        edit.setValidator(validator)
        return edit

    def _install_page(self) -> QWidget:
        page, layout = self._page_shell("Добавление в CUDAynamics", "Просмотрите точный diff. При установке исходные файлы регистрации сохраняются в .cudynamics-builder-backups.")
        path_row = QHBoxLayout()
        self.project_path = QLineEdit(str(locate_default_project(self.root)))
        browse = QPushButton("Папка…"); browse.clicked.connect(self._choose_project)
        preview = QPushButton("Обновить предпросмотр"); preview.clicked.connect(self._prepare)
        path_row.addWidget(QLabel("Проект")); path_row.addWidget(self.project_path, 1); path_row.addWidget(browse); path_row.addWidget(preview)
        layout.addLayout(path_row)
        self.diff_view = QTextEdit(); self.diff_view.setReadOnly(True); self.diff_view.setFont(QFont("Cascadia Mono", 9)); layout.addWidget(self.diff_view, 1)
        row = QHBoxLayout(); row.addStretch()
        self.install_button = QPushButton("Добавить систему в CUDAynamics", objectName="primary"); self.install_button.clicked.connect(self._install); self.install_button.setEnabled(False)
        row.addWidget(self.install_button); layout.addLayout(row)
        return page

    def _set_page(self, index: int) -> None:
        self.pages.setCurrentIndex(index)
        for i, button in enumerate(self.nav): button.setChecked(i == index)

    def _choose_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Система уравнений", "", "Images (*.png *.jpg *.jpeg *.webp *.bmp)")
        if not path: return
        self.image_path = path
        pixmap = QPixmap(path).scaled(self.image_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_preview.setPixmap(pixmap)
        self.statusBar().showMessage(Path(path).name)

    def _update_api_key_visibility(self, backend: str) -> None:
        self.api_key.setVisible(backend != "pix2tex")

    def _recognize(self) -> None:
        if not self.image_path:
            QMessageBox.warning(self, "Нет изображения", "Сначала выберите изображение."); return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.statusBar().showMessage("Распознавание…")
        QApplication.processEvents()
        try:
            text = recognize_image(self.image_path, self.ocr_backend.currentText(), self.api_key.text())
            self.source_text.setPlainText(text)
            self.statusBar().showMessage("OCR завершён. Проверьте текст и нажмите «Разобрать».")
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка OCR", str(exc))
        finally:
            QApplication.restoreOverrideCursor()

    def _parse(self) -> None:
        try:
            self.spec = parse_system(self.source_text.toPlainText(), self.system_id.text(), self.display_name.text())
            self._populate_model(); self._set_page(1)
            self.statusBar().showMessage("Уравнения разобраны; требуется проверка пользователя")
        except ParseError as exc:
            QMessageBox.critical(self, "Не удалось разобрать систему", str(exc))

    def _add_equation_row(self, equation: Equation | None = None) -> None:
        row = self.eq_table.rowCount(); self.eq_table.insertRow(row)
        equation = equation or Equation("x", "0", EquationKind.STATE)
        self.eq_table.setItem(row, 0, QTableWidgetItem(equation.target))
        combo = QComboBox(); combo.addItem("Состояние (интегрировать)", EquationKind.STATE.value); combo.addItem("Алгебраическая", EquationKind.ALGEBRAIC.value)
        combo.setCurrentIndex(0 if equation.kind == EquationKind.STATE else 1); self.eq_table.setCellWidget(row, 1, combo)
        self.eq_table.setItem(row, 2, QTableWidgetItem(equation.expression))

    def _remove_equation_row(self) -> None:
        row = self.eq_table.currentRow()
        if row >= 0: self.eq_table.removeRow(row)

    def _populate_model(self) -> None:
        self.eq_table.setRowCount(0)
        for equation in self.spec.equations: self._add_equation_row(equation)
        self.model_summary.setText(pretty_equations(self.spec))

    def _sync_model(self, quiet: bool = False) -> bool:
        if not self.spec: return False
        equations = []
        try:
            for row in range(self.eq_table.rowCount()):
                target = safe_identifier(self.eq_table.item(row, 0).text().strip())
                expression = self.eq_table.item(row, 2).text().strip()
                kind = EquationKind(self.eq_table.cellWidget(row, 1).currentData())
                equations.append(Equation(target, expression, kind))
            self.spec.equations = equations
            symbols = {str(s) for e in equations for s in sympify_expression(e.expression).free_symbols}
            self.spec.parameters = sorted(symbols - set(self.spec.variables))
            self.spec.ensure_defaults()
            problems = validate_spec(self.spec)
            if problems: raise ValueError("\n".join(problems))
            self.model_summary.setText(f"Интегрируются: {', '.join(self.spec.states)}\nВычисляются: {', '.join(self.spec.algebraic) or '—'}\nПараметры: {', '.join(self.spec.parameters) or '—'}")
            self._populate_defaults()
            if not quiet: QMessageBox.information(self, "Модель корректна", self.model_summary.text())
            return True
        except Exception as exc:
            if not quiet: QMessageBox.critical(self, "Ошибка модели", str(exc))
            return False

    def _populate_defaults(self) -> None:
        if not self.spec: return
        self.variable_table.setRowCount(len(self.spec.variables))
        for row, name in enumerate(self.spec.variables):
            role = "интегрируется" if name in self.spec.states else "вычисляется"
            defaults = self.spec.variable_defaults[name]
            self.variable_table.setItem(row, 0, QTableWidgetItem(name))
            self.variable_table.setItem(row, 1, QTableWidgetItem(role))
            self.variable_table.setCellWidget(row, 2, self._range_combo(defaults.range_kind))
            self.variable_table.setItem(row, 3, QTableWidgetItem(str(defaults.value)))
            self.variable_table.setItem(row, 4, QTableWidgetItem(str(defaults.maximum)))
            self.variable_table.setItem(row, 5, QTableWidgetItem(str(defaults.count)))
        self.parameter_table.setRowCount(len(self.spec.parameters))
        for row, name in enumerate(self.spec.parameters):
            defaults = self.spec.parameter_defaults[name]
            self.parameter_table.setItem(row, 0, QTableWidgetItem(name))
            self.parameter_table.setCellWidget(row, 1, self._range_combo(defaults.range_kind))
            self.parameter_table.setItem(row, 2, QTableWidgetItem(str(defaults.value)))
            self.parameter_table.setItem(row, 3, QTableWidgetItem(str(defaults.maximum)))
            self.parameter_table.setItem(row, 4, QTableWidgetItem(str(defaults.count)))

    @staticmethod
    def _range_combo(range_kind: str) -> QComboBox:
        combo = NoWheelComboBox()
        combo.addItem("Fixed (одно значение)", "Fixed")
        combo.addItem("Count (диапазон)", "Linear")
        combo.setCurrentIndex(0 if range_kind == "Fixed" else 1)
        combo.setToolTip("Fixed — одно значение. Count — указанное число равномерных значений от минимума до максимума.")
        return combo

    def _method_warning(self) -> None:
        notes = []
        if self.method_checks["VariableSymmetryCD"].isChecked(): notes.append("VSCD зависит от порядка строк состояний; линейные неявности решаются формулой, нелинейные — скалярным Ньютоном с аналитической производной.")
        if self.method_checks["ExplicitDormandPrince8"].isChecked(): notes.append("DP8 генерируется с фиксированным шагом, как в текущей библиотеке, без адаптивного контроля ошибки.")
        if self.method_checks["ImplicitEuler"].isChecked() or self.method_checks["ImplicitMidpoint"].isChecked(): notes.append("Неявные методы используют до 8 итераций Ньютона и аналитический Якобиан, вычисленный при генерации.")
        self.method_note.setText(" ".join(notes))

    def _read_settings(self) -> None:
        self.spec.methods = [key for key, check in self.method_checks.items() if check.isChecked()]
        self.spec.step = float(self.step_value.text().replace(",", "."))
        self.spec.steps = int(self.steps_value.text())
        self.spec.transient = int(self.transient_value.text())
        if self.spec.step <= 0: raise ValueError("Шаг h должен быть больше нуля")
        if self.spec.steps < 1: raise ValueError("Число шагов должно быть не меньше 1")
        for row, name in enumerate(self.spec.variables):
            defaults = self.spec.variable_defaults[name]
            defaults.range_kind = self.variable_table.cellWidget(row, 2).currentData()
            defaults.value = float(self.variable_table.item(row, 3).text().replace(",", "."))
            defaults.minimum = defaults.value
            defaults.maximum = float(self.variable_table.item(row, 4).text().replace(",", "."))
            defaults.count = int(self.variable_table.item(row, 5).text())
            self._validate_range(name, defaults)
        for row, name in enumerate(self.spec.parameters):
            defaults = self.spec.parameter_defaults[name]
            defaults.range_kind = self.parameter_table.cellWidget(row, 1).currentData()
            defaults.value = float(self.parameter_table.item(row, 2).text().replace(",", "."))
            defaults.minimum = defaults.value
            defaults.maximum = float(self.parameter_table.item(row, 3).text().replace(",", "."))
            defaults.count = int(self.parameter_table.item(row, 4).text())
            self._validate_range(name, defaults)

    @staticmethod
    def _validate_range(name: str, defaults: AttributeDefaults) -> None:
        if defaults.range_kind == "Linear":
            if defaults.count < 2:
                raise ValueError(f"Для Count-атрибута {name!r} количество должно быть не меньше 2")
            if defaults.maximum < defaults.value:
                raise ValueError(f"Для Count-атрибута {name!r} максимум не может быть меньше минимума")

    def _prepare(self) -> None:
        try:
            if self.spec and self.variable_table.rowCount() == len(self.spec.variables):
                self._read_settings()
            if not self._sync_model(quiet=True): raise ValueError("Сначала исправьте модель на шаге 2")
            problems = validate_spec(self.spec)
            if problems: raise ValueError("\n".join(problems))
            generated = generate(self.spec)
            self.plan = build_plan(self.project_path.text(), self.spec, generated)
            self.diff_view.setPlainText(self.plan.diff)
            self.install_button.setEnabled(True); self._set_page(3)
            self.statusBar().showMessage("План готов; CUDAynamics ещё не изменён")
        except Exception as exc:
            self.plan = None; self.install_button.setEnabled(False)
            QMessageBox.critical(self, "Нельзя подготовить систему", str(exc))

    def _choose_project(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Корень cudaynamics-master", self.project_path.text())
        if path: self.project_path.setText(path)

    def _install(self) -> None:
        if not self.plan: return
        answer = QMessageBox.question(self, "Подтверждение", f"Добавить систему {self.spec.system_id!r} в проект?\nБудут созданы 3 файла и изменены 4 файла регистрации.")
        if answer != QMessageBox.Yes: return
        try:
            backup = apply_plan(self.project_path.text(), self.plan)
            self.install_button.setEnabled(False)
            QMessageBox.information(self, "Готово", f"Система добавлена.\nРезервная копия: {backup}\n\nТеперь пересоберите CUDAynamics в Visual Studio.")
            self.statusBar().showMessage("Система успешно добавлена")
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка установки", str(exc))


def run() -> int:
    app = QApplication.instance() or QApplication([])
    app.setApplicationName("CUDAynamics System Builder")
    window = MainWindow(Path(__file__).resolve().parents[1])
    window.show()
    return app.exec()
