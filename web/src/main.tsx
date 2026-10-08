import "@fontsource/ibm-plex-sans/400.css";
import "@fontsource/ibm-plex-sans/500.css";
import "@fontsource/ibm-plex-sans/600.css";
import "@fontsource/ibm-plex-mono/400.css";
import "@fontsource/ibm-plex-mono/500.css";
import "./styles.css";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { Layout } from "./components/Layout";
import { Evaluation } from "./pages/Evaluation";
import { Method } from "./pages/Method";
import { NotFound } from "./pages/NotFound";
import { RunView } from "./pages/RunView";
import { Runs } from "./pages/Runs";

const root = document.getElementById("root");
if (!root) throw new Error("missing #root element");

createRoot(root).render(
  <StrictMode>
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<Navigate to="/runs" replace />} />
          <Route path="runs" element={<Runs />} />
          <Route path="runs/:runId" element={<RunView tab="network" />} />
          <Route
            path="runs/:runId/process"
            element={<RunView tab="process" />}
          />
          <Route
            path="runs/:runId/citations"
            element={<RunView tab="citations" />}
          />
          <Route
            path="runs/:runId/corrections"
            element={<RunView tab="corrections" />}
          />
          <Route path="evaluation" element={<Evaluation />} />
          <Route path="method" element={<Method />} />
          <Route path="*" element={<NotFound />} />
        </Route>
      </Routes>
    </BrowserRouter>
  </StrictMode>,
);
