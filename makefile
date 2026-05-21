# ─────────────────────────────────────────────
#  Compiler & flags
# ─────────────────────────────────────────────
FC     = gfortran
FFLAGS = -O3 -Wall -fcheck=all

# ─────────────────────────────────────────────
#  Paths
# ─────────────────────────────────────────────
MODDIR  = modfiles
TARGET  = ai_engine
SOURCES = ./linear_regression/logistic_regression.f90 \
          ./polynomial_regression/polynomialreg.f90

# ─────────────────────────────────────────────
#  OS detection  (Windows sets OS=Windows_NT)
# ─────────────────────────────────────────────
ifeq ($(OS),Windows_NT)
    EXT    = .exe
    MKDIR  = if not exist $(MODDIR) mkdir $(MODDIR)
    RM_BIN = if exist $(TARGET)$(EXT) del /f /q $(TARGET)$(EXT)
    RM_MOD = if exist $(MODDIR) rmdir /s /q $(MODDIR)
else
    EXT    =
    MKDIR  = mkdir -p $(MODDIR)
    RM_BIN = rm -f $(TARGET)
    RM_MOD = rm -rf $(MODDIR)
endif

# ─────────────────────────────────────────────
#  Targets
# ─────────────────────────────────────────────
.PHONY: all clean

all: $(TARGET)$(EXT)

$(TARGET)$(EXT): $(SOURCES)
	$(MKDIR)
	$(FC) $(FFLAGS) $(SOURCES) -o $(TARGET)$(EXT) -J$(MODDIR)

clean:
	$(RM_BIN)
	$(RM_MOD)