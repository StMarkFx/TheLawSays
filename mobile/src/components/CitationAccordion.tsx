import { Feather } from "@expo/vector-icons";
import React, { useState } from "react";
import { StyleSheet, Text, TouchableOpacity, View } from "react-native";

import type { Chunk } from "../lib/types";
import { colors, fonts } from "../theme";

interface CitationAccordionProps {
  chunks: Chunk[];
}

export function CitationAccordion({ chunks }: CitationAccordionProps) {
  const [openIds, setOpenIds] = useState<Record<number, boolean>>({});

  const toggle = (id: number) => {
    setOpenIds((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Citations</Text>
      {chunks.map((chunk) => {
        const isOpen = !!openIds[chunk.id];
        return (
          <View key={`${chunk.id}-${chunk.source}`} style={styles.card}>
            <TouchableOpacity
              onPress={() => toggle(chunk.id)}
              style={styles.header}
              accessibilityLabel="Toggle citation"
            >
              <View style={styles.headerText}>
                <Text style={styles.source}>{chunk.source}</Text>
                <Text style={styles.jurisdiction}>{chunk.jurisdiction}</Text>
              </View>
              <Feather name={isOpen ? "chevron-up" : "chevron-down"} size={18} color={colors.muted} />
            </TouchableOpacity>
            {isOpen && (
              <View style={styles.body}>
                <Text style={styles.bodyText}>{chunk.text}</Text>
              </View>
            )}
          </View>
        );
      })}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    paddingTop: 8
  },
  title: {
    color: colors.muted,
    fontSize: 12,
    textTransform: "uppercase",
    letterSpacing: 2,
    fontFamily: fonts.medium,
    marginBottom: 12
  },
  card: {
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: 16,
    backgroundColor: colors.surface,
    marginBottom: 12
  },
  header: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingHorizontal: 14,
    paddingVertical: 12
  },
  headerText: {
    flex: 1,
    marginRight: 8
  },
  source: {
    color: colors.foreground,
    fontFamily: fonts.medium,
    fontSize: 14
  },
  jurisdiction: {
    color: colors.muted,
    fontSize: 12,
    marginTop: 4,
    fontFamily: fonts.regular
  },
  body: {
    paddingHorizontal: 14,
    paddingBottom: 12
  },
  bodyText: {
    color: colors.foreground,
    fontSize: 14,
    lineHeight: 20,
    fontFamily: fonts.regular
  }
});
